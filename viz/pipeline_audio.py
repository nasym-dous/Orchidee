import threading
from queue import Queue
from .config import AppConfig
from .io_audio import load_audio_stereo_ffmpeg
from .stats import Timer
from .types import AudioChunk


def start_audio_source(cfg: AppConfig, output: Queue, stop_token: object) -> threading.Thread:
    """Decode audio and emit a single AudioChunk downstream."""

    def _run():
        with Timer("audio decode"):
            audio, sr = load_audio_stereo_ffmpeg(cfg.audio.audio_path, cfg.audio.target_sr, verbose=cfg.verbose)
        output.put(AudioChunk(samples=audio, sample_rate=sr))
        output.put(stop_token)
        if cfg.verbose:
            print(f"🎵 sr={sr} | samples={audio.shape[0]} | duration={audio.shape[0]/sr:.2f}s")

    t = threading.Thread(target=_run, name="audio_source", daemon=True)
    t.start()
    return t
