"""Track a textured, rigid planar selection without adding fiducial markers."""

from __future__ import annotations

from typing import NoReturn

import cv2
import numpy as np


class TrackingLost(RuntimeError):
    """The selection can no longer be located reliably; reselect to resume."""


class RegionTracker:
    MIN_POINTS = 8

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.gray: np.ndarray | None = None
        self.points: np.ndarray | None = None
        self.anchors: np.ndarray | None = None
        self.reference_polygon: np.ndarray | None = None
        self.polygon: np.ndarray | None = None

    @property
    def active(self) -> bool:
        return self.gray is not None

    @staticmethod
    def _gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame.copy()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _lost(self, message: str) -> NoReturn:
        self.reset()
        raise TrackingLost(message)

    def start(self, frame: np.ndarray, polygon: np.ndarray) -> np.ndarray:
        self.reset()
        gray = self._gray(frame)
        poly = np.asarray(polygon, dtype=np.float32)
        if (poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3
                or not np.isfinite(poly).all()
                or abs(cv2.contourArea(poly)) < 100):
            self._lost("invalid selection; draw a larger region")
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [np.rint(poly).astype(np.int32)], 255)
        # Avoid the object/background boundary and the projected outline.
        mask = cv2.erode(mask, np.ones((7, 7), dtype=np.uint8))
        points = cv2.goodFeaturesToTrack(
            gray, maxCorners=250, qualityLevel=0.02, minDistance=7,
            mask=mask, blockSize=7,
        )
        if points is None or len(points) < self.MIN_POINTS:
            self._lost("too little texture; select a textured part of the object")
        self.gray = gray
        self.points = points
        self.anchors = points.copy()
        self.reference_polygon = poly.copy()
        self.polygon = poly.copy()
        return self.polygon.copy()

    def update(self, frame: np.ndarray) -> np.ndarray:
        if (self.gray is None or self.points is None or self.anchors is None
                or self.reference_polygon is None):
            raise TrackingLost("no active selection; draw a region")
        gray = self._gray(frame)
        if gray.shape != self.gray.shape:
            self._lost("camera dimensions changed; draw again")
        params = dict(
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        current, forward, _ = cv2.calcOpticalFlowPyrLK(
            self.gray, gray, self.points, None, **params,
        )
        if current is None or forward is None or not np.isfinite(current).all():
            self._lost("optical flow failed; draw again")
        backward, reverse, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.gray, current, None, **params,
        )
        if backward is None or reverse is None:
            self._lost("optical flow verification failed; draw again")
        xy = current.reshape(-1, 2)
        fb_error = np.linalg.norm((backward - self.points).reshape(-1, 2), axis=1)
        valid = (
            (forward.ravel() != 0) & (reverse.ravel() != 0)
            & np.isfinite(fb_error) & (fb_error < 1.5)
            & (xy[:, 0] >= 0) & (xy[:, 0] < gray.shape[1])
            & (xy[:, 1] >= 0) & (xy[:, 1] < gray.shape[0])
        )
        if int(valid.sum()) < self.MIN_POINTS:
            self._lost("not enough tracked features; draw again")
        anchors = self.anchors[valid]
        current = current[valid]
        # Fit against the original anchors, rather than accumulating transforms.
        transform, inliers = cv2.estimateAffinePartial2D(
            anchors, current, method=cv2.RANSAC,
            ransacReprojThreshold=2.0, maxIters=2000, confidence=0.99,
        )
        if transform is None or inliers is None or not np.isfinite(transform).all():
            self._lost("motion estimate failed; draw again")
        keep = inliers.ravel() != 0
        if int(keep.sum()) < self.MIN_POINTS or float(keep.mean()) < 0.6:
            self._lost("inconsistent object motion; draw again")
        scale = float(np.hypot(transform[0, 0], transform[1, 0]))
        if not 0.9 <= scale <= 1.1:
            self._lost("depth or tilt changed too much for planar tracking")
        polygon = cv2.transform(
            self.reference_polygon.reshape(-1, 1, 2), transform,
        ).reshape(-1, 2)
        if (not np.isfinite(polygon).all()
                or (polygon[:, 0] < 0).any() or (polygon[:, 0] >= gray.shape[1]).any()
                or (polygon[:, 1] < 0).any() or (polygon[:, 1] >= gray.shape[0]).any()):
            self._lost("selection left the camera view; draw again")
        self.gray = gray
        self.points = current[keep].copy()
        self.anchors = anchors[keep].copy()
        self.polygon = polygon
        return polygon.copy()
