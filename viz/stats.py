import time
import os
import resource

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
