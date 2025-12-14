import threading
from queue import Queue

from .config import AppConfig
from .renderer_spectrogram import SpectrogramRenderer
from .types import AlphaBatch, AudioChunk


def start_spectrogram_filter(cfg: AppConfig, audio_in: Queue, alpha_out: Queue, stop_token: object) -> threading.Thread:
    """Convert decoded audio into batches of spectrogram alpha masks."""

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
                alphas = renderer.next_alphas(t, n)

                alpha_out.put(AlphaBatch(start_frame=t, alphas=alphas))
                t += n

            audio_in.task_done()

    t = threading.Thread(target=_run, name="spectrogram_filter", daemon=True)
    t.start()
    return t
