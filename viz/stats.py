import time
import os
import resource
from queue import Queue
from typing import Dict, Iterable, List, Union

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


def _bar(fraction: float, width: int = 16) -> str:
    fraction = float(np.clip(fraction, 0.0, 1.0))
    filled = int(round(fraction * width))
    return "█" * filled + "░" * (width - filled)


def format_batch_telemetry(
    stage: str,
    start_frame: int,
    batch_len: int,
    batch_bytes: int,
    q: Queue,
    fps: float,
    detail: str | None = None,
    suffix: str = "",
) -> str:
    """Return a compact, single-line telemetry summary with bars.

    The returned string is intended to be rendered on a live dashboard rather than
    appended as log spam. It includes a small bar for queue pressure and optional
    detail about the underlying worker (ffmpeg, compositor backend, etc.).
    """

    queued_mb = queue_buffer_mb(q, batch_bytes)
    batch_mb = bytes_mb(batch_bytes)

    capacity = q.maxsize if q.maxsize > 0 else max(q.qsize(), 1)
    frac = q.qsize() / capacity if capacity else 0.0
    pressure_bar = _bar(frac)

    detail_str = f"[{detail}]" if detail else ""

    return (
        f"{stage:<24} {detail_str:<20} | start={start_frame:<5} n={batch_len:<3} "
        f"batch≈{batch_mb:5.2f}MB | queued={q.qsize():<3}/{capacity:<3} {pressure_bar} ({queued_mb:6.2f}MB) "
        f"| rate≈{fps:5.1f} fps{suffix}"
    )


class _PerfBoard:
    """Maintain and redraw in-place performance bars for pipeline stages."""

    def __init__(self):
        self._lines: Dict[str, str] = {}
        self._order: List[str] = []
        self._rendered = 0

    def update(self, stage: str, line: str):
        if stage not in self._lines:
            self._order.append(stage)
        self._lines[stage] = line
        self._render()

    def _render(self):
        if self._rendered:
            # move cursor up to redraw in-place
            print("\033[F" * self._rendered, end="")

        for stage in self._order:
            print("\033[K" + self._lines[stage])

        self._rendered = len(self._order)


perf_board = _PerfBoard()
