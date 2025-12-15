from __future__ import annotations

import sys
import threading
import time
import os
import resource
from queue import Queue
from typing import Iterable, Union

import numpy as np

try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False

if _HAS_PSUTIL:
    _proc = psutil.Process(os.getpid())
    def ram_mb():
        return _proc.memory_info().rss / (1024 ** 2)
else:
    def ram_mb():
        # macOS: ru_maxrss is KB
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


class Timer:
    def __init__(self, name): self.name = name
    def __enter__(self): self.t0 = time.perf_counter()
    def __exit__(self, *_):
        dt = time.perf_counter() - self.t0
        print(f"⏱️ {self.name}: {dt:.3f}s")


class PerfCounter:
    def __init__(self):
        self.frames = 0
        self.t0 = None
        self.t1 = None

    def start(self):
        self.t0 = time.perf_counter()

    def tick(self, n=1):
        self.frames += n

    def stop(self):
        self.t1 = time.perf_counter()

    def avg_fps(self):
        if self.t0 is None or self.t1 is None:
            return 0.0
        return self.frames / max(self.t1 - self.t0, 1e-9)


MB = 1024 ** 2


def bytes_mb(num_bytes: int) -> float:
    return num_bytes / MB


def _nbytes(obj: Union[np.ndarray, Iterable[np.ndarray]]) -> int:
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj, Iterable):
        return sum(_nbytes(x) for x in obj)
    return 0


def batch_memory_mb(batch_content: Union[np.ndarray, Iterable[np.ndarray]]) -> float:
    """Return the memory footprint of a batch in megabytes."""

    return bytes_mb(_nbytes(batch_content))


def queue_buffer_mb(q: Queue, batch_bytes: int) -> float:
    """Approximate buffered megabytes for a queue of equally-sized batches."""

    return bytes_mb(q.qsize() * batch_bytes)


def _bar(progress: float, width: int = 14) -> str:
    filled = int(max(min(progress, 1.0), 0.0) * width)
    return "[" + ("█" * filled) + ("░" * (width - filled)) + "]"


def _bar_from_value(value: float, hint: float, width: int = 14) -> str:
    if hint <= 0:
        return _bar(0.0, width)
    return _bar(min(value / hint, 1.0), width)


def format_batch_telemetry(
    stage: str,
    start_frame: int,
    batch_len: int,
    batch_bytes: int,
    q: Queue,
    fps: float,
    *,
    target_fps: float | None = None,
    queue_hint: int = 8,
    engine: str | None = None,
    extra: str | None = None,
) -> str:
    """Return a single-line status string summarizing the batch."""

    queued_mb = queue_buffer_mb(q, batch_bytes)
    batch_mb = bytes_mb(batch_bytes)

    fps_hint = target_fps if target_fps and target_fps > 0 else fps
    queue_hint = max(queue_hint, 1)

    fps_bar = _bar_from_value(fps, fps_hint, width=12)
    queue_bar = _bar_from_value(q.qsize(), queue_hint, width=10)
    batch_bar = _bar_from_value(batch_mb, batch_mb * 2 if batch_mb > 0 else 1.0, width=10)

    engine_suffix = f" | engine={engine}" if engine else ""
    extra_suffix = f" | {extra}" if extra else ""

    return (
        f"{stage:<24} start {start_frame:<6} n={batch_len:<3} "
        f"fps {fps_bar} {fps:5.1f} | batch {batch_bar} {batch_mb:6.2f} MB "
        f"| queued {queue_bar} {q.qsize():>2} ({queued_mb:5.2f} MB)"
        f"{engine_suffix}{extra_suffix}"
    )


class TelemetryBoard:
    """Render telemetry as in-place bars instead of log spam."""

    def __init__(self):
        self._lock = threading.Lock()
        self._lines: dict[str, str] = {}
        self._order: list[str] = []
        self._rendered = 0

    def update(self, stage: str, line: str):
        with self._lock:
            if stage not in self._order:
                self._order.append(stage)
            self._lines[stage] = line
            self._render()

    def _render(self):
        if self._rendered:
            sys.stdout.write(f"\x1b[{self._rendered}F")
        for stage in self._order:
            sys.stdout.write("\x1b[2K" + self._lines[stage] + "\n")
        sys.stdout.flush()
        self._rendered = len(self._order)


telemetry_board = TelemetryBoard()


def log_batch_telemetry(**kwargs):
    """Helper used by pipeline stages to refresh the telemetry board."""

    stage = kwargs.get("stage", "stage")
    telemetry_board.update(stage, format_batch_telemetry(**kwargs))
