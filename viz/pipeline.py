import threading
import queue
import time
import cv2
from typing import Callable
import numpy as np
from .config import AppConfig
from .stats import PerfCounter, ram_mb


CompositorFn = Callable[[np.ndarray, np.ndarray, AppConfig], np.ndarray]


def run_pipeline(
    cfg: AppConfig,
    renderer,
    cover_bgr: np.ndarray,
    compositor: CompositorFn,
) -> str:
    q = queue.Queue(cfg.render.max_buffer_batches)
    STOP = object()

    perf = PerfCounter()

    def producer():
        if cfg.verbose:
            print("🚀 Producer started")
        t = 0
        while t < renderer.n_frames:
            n = min(cfg.render.batch, renderer.n_frames - t)

            t0 = time.perf_counter()
            alphas = renderer.next_alphas(t, n)
            dt = time.perf_counter() - t0

            if cfg.verbose and (t == 0 or t % (cfg.video.fps * 5) == 0):
                fps_prod = n / max(dt, 1e-6)
                print(f"🧠 Producer batch {n}: {dt:.4f}s => {fps_prod:.1f} fps (producer)")

            q.put(alphas)
            t += n
        q.put(STOP)

    def consumer():
        out = cv2.VideoWriter(
            cfg.paths.out_avi,
            cv2.VideoWriter_fourcc(*cfg.video.fourcc),
            cfg.video.fps,
            (cfg.video.w, cfg.video.h)
        )
        if not out.isOpened():
            raise RuntimeError("VideoWriter non ouvert")

        perf.start()
        written = 0

        while True:
            item = q.get()
            if item is STOP:
                break

            for alpha_small in item:
                frame = compositor(cover_bgr, alpha_small, cfg)
                out.write(frame)
                written += 1
                perf.tick(1)

                if cfg.verbose and written % (cfg.video.fps * 5) == 0:
                    avg_fps = perf.frames / max(time.perf_counter() - perf.t0, 1e-6)
                    print(f"📼 {written} frames | avg FPS ≈ {avg_fps:.1f} | RAM ≈ {ram_mb():.0f} MB")

            q.task_done()

        out.release()
        perf.stop()

    tp = threading.Thread(target=producer, name="producer")
    tc = threading.Thread(target=consumer, name="consumer")
    tc.start(); tp.start()
    tp.join(); tc.join()

    if cfg.verbose:
        print("🚀 Render performance")
        print(f"  frames rendered : {perf.frames}")
        print(f"  avg FPS         : {perf.avg_fps():.2f}")
        print(f"  RAM (now)       : {ram_mb():.0f} MB")
        print(f"✅ Vidéo MJPG terminée : {cfg.paths.out_avi}")

    return cfg.paths.out_avi
