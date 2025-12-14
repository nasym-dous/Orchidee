import subprocess


def mux_audio(in_video: str, audio_path: str, out_mp4: str, verbose: bool, copy_video: bool):
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        in_video,
        "-i",
        audio_path,
    ]

    if copy_video:
        cmd.extend(["-c:v", "copy"])
    else:
        cmd.extend([
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-crf",
            "18",
        ])

    cmd.extend([
        "-c:a",
        "aac",
        "-shortest",
        out_mp4,
    ])

    if not verbose:
        cmd.insert(1, "-v")
        cmd.insert(2, "error")

    subprocess.run(cmd, check=True)
