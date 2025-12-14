import queue
from queue import Queue
import threading
import numpy as np
from .config import AppConfig
from .compositor import compose_frame_from_alpha
from .pipeline_audio import start_audio_source
from .pipeline_renderer import start_spectrogram_filter
from .pipeline_formants import start_formant_trace_filter
from .pipeline_compositor import start_compositor_filter
from .pipeline_telemetry import start_telemetry_filter
from .pipeline_encoder import start_encoder_sink
from .types import AudioChunk, AlphaBatch, FrameBatch


TypedQueue = Queue[AudioChunk | AlphaBatch | FrameBatch | object]


def run_pipeline(cfg: AppConfig, cover_bgr: np.ndarray) -> str:
    """Assemble the filters into a linear pipeline.

    source -> renderer -> compositor -> encoder
    """
    stop_token = object()

    audio_q: TypedQueue = queue.Queue(maxsize=1)
    meter_audio_q: TypedQueue = queue.Queue(maxsize=1)
    formant_audio_q: TypedQueue = queue.Queue(maxsize=1)
    spectrogram_alpha_q: TypedQueue = queue.Queue(maxsize=cfg.render.max_buffer_batches)
    formant_alpha_q: TypedQueue = queue.Queue(maxsize=cfg.render.max_buffer_batches)
    alpha_q: TypedQueue = queue.Queue(maxsize=cfg.render.max_buffer_batches)
    frame_q: TypedQueue = queue.Queue(maxsize=cfg.render.max_buffer_batches)

    threads: list[threading.Thread] = [
        start_encoder_sink(cfg, frame_q, stop_token),
        start_compositor_filter(cfg, cover_bgr, compose_frame_from_alpha, alpha_q, frame_q, stop_token),
        start_telemetry_filter(cfg, meter_audio_q, formant_alpha_q, alpha_q, stop_token),
        start_formant_trace_filter(cfg, formant_audio_q, spectrogram_alpha_q, formant_alpha_q, stop_token),
        start_spectrogram_filter(cfg, audio_q, spectrogram_alpha_q, stop_token),
        # start_renderer_filter(cfg, audio_q, alpha_q, stop_token),  # scrolling filter disabled in favor of spectrogram
        start_audio_source(cfg, [audio_q, meter_audio_q, formant_audio_q], stop_token),
    ]

    for t in threads:
        t.join()

    return cfg.paths.out_final
