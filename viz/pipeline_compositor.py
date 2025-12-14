import threading
from queue import Queue
from typing import Callable
import numpy as np
from .config import AppConfig
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
            frames = [compositor(cover_bgr, alpha, cfg) for alpha in alpha_batch.alphas]
            frame_out.put(FrameBatch(start_frame=alpha_batch.start_frame, frames=frames))
            alpha_in.task_done()

    t = threading.Thread(target=_run, name="compositor_filter", daemon=True)
    t.start()
    return t
