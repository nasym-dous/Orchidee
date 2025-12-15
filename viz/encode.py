import subprocess

from .config import EncodeConfig


def mux_audio(in_avi: str, audio_path: str, out_mp4: str, verbose: bool, encode: EncodeConfig):
    cmd = [
        "ffmpeg", "-y",
        "-i", in_avi,
        "-i", audio_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", encode.video_preset,
        "-crf", str(encode.video_crf),
        "-c:a", encode.audio_codec,
        "-b:a", encode.audio_bitrate,
        "-shortest",
        out_mp4
    ]
    subprocess.run(cmd, check=True)
