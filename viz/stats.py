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


def format_batch_telemetry(stage: str, start_frame: int, batch_len: int, batch_bytes: int, q: Queue, fps: float) -> str:
    queued_mb = queue_buffer_mb(q, batch_bytes)
    batch_mb = bytes_mb(batch_bytes)
    return (
        f"{stage:<24} \tstart = {start_frame:<6} n = {batch_len}, "
        f"batch ≈ {batch_mb:.2f} MB, queued ≈ {q.qsize()} ({queued_mb:.2f} MB), "
        f"\trate ≈ {fps:.1f} fps"
    )
