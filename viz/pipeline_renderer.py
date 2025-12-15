import threading
import time
from queue import Queue
from .config import AppConfig
from .renderer_scrolling import ScrollingRenderer
from .renderer_spectrogram import SpectrogramRenderer
from .stats import log_batch_telemetry
from .types import AlphaBatch, AudioChunk


def start_renderer_filter(cfg: AppConfig, audio_in: Queue, alpha_out: Queue, stop_token: object) -> threading.Thread:
    """Convert decoded audio into batches of alpha masks."""

    def _run():
        while True:
            item = audio_in.get()
            if item is stop_token:
                audio_in.task_done()
                alpha_out.put(stop_token)
                break

            chunk: AudioChunk = item
            renderer = ScrollingRenderer(chunk.samples, cfg)

            t = 0
            while t < renderer.n_frames:
                n = min(cfg.render.batch, renderer.n_frames - t)
                t0 = time.perf_counter()
                alphas = renderer.next_alphas(t, n)
                dt = time.perf_counter() - t0

                if cfg.verbose and (t == 0 or t % (cfg.video.fps * 5) == 0):
                    fps_prod = n / max(dt, 1e-6)
                    batch_bytes = int(alphas.nbytes)
                    log_batch_telemetry(
                        stage="🧠 Renderer (producer)",
                        start_frame=t,
                        batch_len=n,
                        batch_bytes=batch_bytes,
                        q=alpha_out,
                        fps=fps_prod,
                        target_fps=cfg.video.fps,
                        engine="numpy scrolling",
                    )

                alpha_out.put(AlphaBatch(start_frame=t, alphas=alphas))
                t += n

            audio_in.task_done()

    t = threading.Thread(target=_run, name="renderer_filter", daemon=True)
    t.start()
    return t


def start_spectrogram_filter(cfg: AppConfig, audio_in: Queue, alpha_out: Queue, stop_token: object) -> threading.Thread:
    """Convert decoded audio into spectrogram-style alpha batches."""

    def _run():
        while True:
            item = audio_in.get()
            if item is stop_token:
                audio_in.task_done()
                alpha_out.put(stop_token)
                break

            chunk: AudioChunk = item
            renderer = SpectrogramRenderer(chunk.samples, cfg)

            t = 0
            while t < renderer.n_frames:
                n = min(cfg.render.batch, renderer.n_frames - t)
                t0 = time.perf_counter()
                alphas = renderer.next_alphas(t, n)
                dt = time.perf_counter() - t0

                if cfg.verbose and (t == 0 or t % (cfg.video.fps * 5) == 0):
                    fps_prod = n / max(dt, 1e-6)
                    batch_bytes = int(alphas.nbytes)
                    log_batch_telemetry(
                        stage="🧠 Spectrogram (producer)",
                        start_frame=t,
                        batch_len=n,
                        batch_bytes=batch_bytes,
                        q=alpha_out,
                        fps=fps_prod,
                        target_fps=cfg.video.fps,
                        engine="numpy spectrogram",
                    )

                alpha_out.put(AlphaBatch(start_frame=t, alphas=alphas))
                t += n

            audio_in.task_done()

    t = threading.Thread(target=_run, name="spectrogram_filter", daemon=True)
    t.start()
    return t
