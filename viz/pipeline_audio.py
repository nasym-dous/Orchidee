import threading
from collections.abc import Iterable
from queue import Queue
from .config import AppConfig
from .io_audio import load_audio_stereo_ffmpeg
from .stats import Timer
from .types import AudioChunk


def _normalize_outputs(output: Queue | Iterable[Queue]) -> list[Queue]:
    if isinstance(output, Queue):
        return [output]
    if isinstance(output, Iterable):
        return list(output)
    return [output]


def start_audio_source(cfg: AppConfig, output: Queue | Iterable[Queue], stop_token: object) -> threading.Thread:
    """Decode audio and emit a single AudioChunk downstream to one or more queues."""

    def _run():
        outputs = _normalize_outputs(output)
        max_seconds = cfg.audio.clip_seconds if cfg.audio.clip_audio else None
        with Timer("audio decode"):
            audio, sr = load_audio_stereo_ffmpeg(
                cfg.audio.audio_path,
                cfg.audio.target_sr,
                verbose=cfg.verbose,
                max_seconds=max_seconds,
            )

        for q in outputs:
            q.put(AudioChunk(samples=audio, sample_rate=sr))
            q.put(stop_token)

        if cfg.verbose:
            print(
                f"🎵 sr={sr} | samples={audio.shape[0]} | duration={audio.shape[0]/sr:.2f}s"
                f" | clip={max_seconds}s → outputs={len(outputs)}"
            )

    t = threading.Thread(target=_run, name="audio_source", daemon=True)
    t.start()
    return t
