import threading
import time
from queue import Queue
from typing import Callable
import numpy as np
from .config import AppConfig
from .stats import batch_memory_mb, log_batch_telemetry
from .types import AlphaBatch, FrameBatch

CompositorFn = Callable[[np.ndarray, np.ndarray, AppConfig], np.ndarray]


def start_compositor_filter(
    cfg: AppConfig,
    cover_bgr: np.ndarray,
    compositor: CompositorFn,
    alpha_in: Queue,
    frame_out: Queue,
    stop_token: object,
) -> threading.Thread:
    """Transform alpha batches into BGR frames ready for encoding."""

    def _run():
        while True:
            item = alpha_in.get()
            if item is stop_token:
                alpha_in.task_done()
                frame_out.put(stop_token)
                break

            alpha_batch: AlphaBatch = item
            t0 = time.perf_counter()
            frames = [compositor(cover_bgr, alpha, cfg) for alpha in alpha_batch.alphas]
            dt = time.perf_counter() - t0
            if cfg.verbose and (alpha_batch.start_frame == 0 or alpha_batch.start_frame % (cfg.video.fps * 5) == 0):
                fps_cons = len(frames) / max(dt, 1e-6)
                alpha_bytes = int(alpha_batch.alphas.nbytes)
                frame_mb = batch_memory_mb(frames)
                log_batch_telemetry(
                    stage="🖼️ Compositor (consumer)",
                    start_frame=alpha_batch.start_frame,
                    batch_len=len(frames),
                    batch_bytes=alpha_bytes,
                    q=alpha_in,
                    fps=fps_cons,
                    target_fps=cfg.video.fps,
                    engine="NumPy compositor",
                    extra=f"output≈{frame_mb:.2f} MB",
                )
            frame_out.put(FrameBatch(start_frame=alpha_batch.start_frame, frames=frames))
            alpha_in.task_done()

    t = threading.Thread(target=_run, name="compositor_filter", daemon=True)
    t.start()
    return t
