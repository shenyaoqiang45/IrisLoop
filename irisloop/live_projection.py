"""Serialize slow projector uploads, keeping only the newest pending mask.

A full 640x480 1bpp frame takes seconds over BLE while a camera delivers tens
of frames per second. Callers ``submit()`` a mask per camera frame and call
``pump()``; at most one upload runs at a time and stale positions are replaced
instead of queued. Do not cancel an in-flight transfer: interrupting a file
transfer and interleaving another upload wedges the firmware.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import numpy as np

from irisloop.image_pack import HEIGHT, WIDTH


class LatestMask:
    """Drop-stale coalescer around an async ``send(mask)`` callable."""

    def __init__(self, send: Callable[[np.ndarray], Awaitable[None]]) -> None:
        self._sender = send
        self.pending: np.ndarray | None = None
        self.last = np.zeros((HEIGHT, WIDTH), np.uint8)
        self.task: asyncio.Task[None] | None = None
        self.uploads = 0

    def submit(self, mask: np.ndarray) -> None:
        self.pending = mask.copy()

    async def _upload(self, mask: np.ndarray) -> None:
        await self._sender(mask)
        self.last = mask
        self.uploads += 1

    def pump(self) -> None:
        if self.task is not None:
            if not self.task.done():
                return
            self.task.result()  # surface upload failures instead of swallowing them
            self.task = None
        if self.pending is not None:
            mask, self.pending = self.pending, None
            if not np.array_equal(mask, self.last):
                self.task = asyncio.create_task(self._upload(mask))

    async def finish(self) -> None:
        self.pending = None
        if self.task is not None:
            await self.task
