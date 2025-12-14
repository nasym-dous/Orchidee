import subprocess


def mux_audio(in_avi: str, audio_path: str, out_mp4: str, verbose: bool):
    cmd = [
        "ffmpeg", "-y",
        "-i", in_avi,
        "-i", audio_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "18",
        "-c:a", "aac",
        "-shortest",
        out_mp4
    ]
    if not verbose:
        cmd.insert(1, "-v")
        cmd.insert(2, "error")
    subprocess.run(cmd, check=True)
