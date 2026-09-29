"""Persisted camera-space targets for repeatable physical projection tests."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from irisloop.calibrate import Calibration
from irisloop.image_pack import HEIGHT, WIDTH

SCHEMA = "irisloop.target_path/1"


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class TargetPath:
    scene: str
    mode: str
    camera_size: list[int]
    paths: list[list[list[float]]]
    calibration_sha256: str
    created: str
    note: str = ""
    schema: str = SCHEMA

    @classmethod
    def create(
        cls,
        paths,
        *,
        scene: str,
        mode: str,
        camera_size,
        calibration_path: str,
        note: str = "",
    ) -> "TargetPath":
        if mode not in ("stroke", "shape"):
            raise ValueError(f"unsupported target mode: {mode}")
        normalized = [
            np.asarray(path, dtype=np.float64).reshape(-1, 2).tolist()
            for path in paths
        ]
        if not normalized or any(len(path) < 2 for path in normalized):
            raise ValueError("target must contain at least one path with two points")
        return cls(
            scene=scene,
            mode=mode,
            camera_size=[int(camera_size[0]), int(camera_size[1])],
            paths=normalized,
            calibration_sha256=file_sha256(calibration_path),
            created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            note=note,
        )

    def arrays(self) -> list[np.ndarray]:
        return [np.asarray(path, dtype=np.float64) for path in self.paths]

    def validate_calibration(self, calibration_path: str) -> None:
        actual = file_sha256(calibration_path)
        if actual != self.calibration_sha256:
            raise ValueError(
                "target path belongs to a different calibration "
                f"(expected {self.calibration_sha256[:12]}, got {actual[:12]})"
            )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(asdict(self), stream, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "TargetPath":
        with open(path, encoding="utf-8") as stream:
            data = json.load(stream)
        if data.get("schema") != SCHEMA:
            raise ValueError(f"unsupported target path schema: {data.get('schema')}")
        return cls(**data)


@dataclass
class TargetRender:
    mask: np.ndarray
    camera_quad: np.ndarray
    projector_quad: np.ndarray
    requested_width_px: int
    mask_pixels: int


def _aspect_fit(quad, content_aspect: float) -> np.ndarray:
    quad = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    x0, y0 = quad[0]
    x1, y1 = quad[2]
    width, height = x1 - x0, y1 - y0
    if width <= 0 or height <= 0:
        raise ValueError("target has an empty bounding box")
    if width / height > content_aspect:
        fitted = height * content_aspect
        x0 += (width - fitted) / 2
        x1 = x0 + fitted
    else:
        fitted = width / content_aspect
        y0 += (height - fitted) / 2
        y1 = y0 + fitted
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def render_target(
    target: TargetPath,
    calibration: Calibration,
    width_px: int,
    *,
    content_aspect: float = WIDTH / HEIGHT,
) -> TargetRender:
    if width_px < 1:
        raise ValueError("width_px must be >= 1")
    camera_width, camera_height = target.camera_size
    paths = target.arrays()
    points = np.concatenate(paths).reshape(-1, 2)
    mapped = calibration.cam_to_proj(points)
    if not np.isfinite(mapped).all() or not all(
        0 <= x < WIDTH and 0 <= y < HEIGHT for x, y in mapped
    ):
        raise ValueError("part of the target is outside the projection field")

    camera_mask = np.zeros((camera_height, camera_width), dtype=np.uint8)
    if target.mode == "stroke":
        for path in paths:
            cv2.polylines(
                camera_mask,
                [np.rint(path).astype(np.int32)],
                False,
                1,
                width_px,
                lineType=cv2.LINE_8,
            )
    else:
        cv2.fillPoly(
            camera_mask,
            [np.rint(paths[0]).astype(np.int32)],
            1,
        )

    ys, xs = np.nonzero(camera_mask)
    if len(xs) == 0:
        raise ValueError("target renders to an empty mask")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    camera_quad = _aspect_fit(
        [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
        content_aspect,
    )
    projector_quad = calibration.cam_to_proj(camera_quad)
    if not all(0 <= x < WIDTH and 0 <= y < HEIGHT for x, y in projector_quad):
        raise ValueError("target bounding box is outside the projection field")

    mask = cv2.warpPerspective(
        camera_mask,
        calibration.H_inv,
        (WIDTH, HEIGHT),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask = (mask > 0).astype(np.uint8)
    return TargetRender(
        mask=mask,
        camera_quad=camera_quad,
        projector_quad=projector_quad,
        requested_width_px=width_px,
        mask_pixels=int(mask.sum()),
    )
