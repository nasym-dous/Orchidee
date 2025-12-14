import subprocess
import numpy as np


def load_audio_stereo_ffmpeg(path: str, target_sr: int, verbose: bool):
    cmd = [
        "ffmpeg", "-v", "info" if verbose else "error",
        "-i", path,
        "-vn",
        "-ac", "2",
        "-ar", str(target_sr),
        "-f", "f32le",
        "pipe:1"
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, check=True)
    audio = np.frombuffer(p.stdout, dtype=np.float32).reshape((-1, 2))
    return audio, target_sr
