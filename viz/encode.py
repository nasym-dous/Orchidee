import subprocess


def mux_audio(in_avi: str, audio_path: str, out_mp4: str, verbose: bool, max_audio_seconds: int | None):
    cmd = ["ffmpeg", "-y"]
    if not verbose:
        cmd.extend(["-v", "error"])
    if max_audio_seconds is not None:
        cmd.extend(["-t", str(max_audio_seconds)])
    cmd.extend([
        "-i", in_avi,
        "-i", audio_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "18",
        "-c:a", "aac",
        "-shortest",
        out_mp4
    ])
    subprocess.run(cmd, check=True)
