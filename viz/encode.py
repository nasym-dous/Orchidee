import subprocess

from .config import EncodeConfig


def mux_audio(
    in_video: str,
    audio_path: str,
    out_mp4: str,
    verbose: bool,
    verbose_lib: bool,
    encode: EncodeConfig,
):
    loglevel = "info" if verbose_lib else "error"
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        loglevel,
        "-i",
        in_video,
        "-i",
        audio_path,
        "-c:v",
        "copy",
        "-c:a",
        encode.audio_codec,
        "-b:a",
        encode.audio_bitrate,
        "-shortest",
        out_mp4,
    ]
    subprocess.run(cmd, check=True)
