import subprocess
import threading
import time
from queue import Queue

from .config import AppConfig
from .encode import mux_audio
from .stats import PerfCounter, batch_memory_mb, format_batch_telemetry, ram_mb
from .types import FrameBatch


def start_encoder_sink(cfg: AppConfig, frames_in: Queue, stop_token: object) -> threading.Thread:
    """Encode frames to disk and mux the original audio."""

    def _run():
        loglevel = "info" if cfg.verbose_lib else "error"
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            loglevel,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{cfg.video.w}x{cfg.video.h}",
            "-r",
            str(cfg.video.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "h264_videotoolbox",
            "-pix_fmt",
            "yuv420p",
            "-b:v",
            cfg.encode.video_bitrate,
            cfg.paths.out_video,
        ]

        try:
            proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg introuvable dans le PATH") from exc
        if proc.stdin is None:
            raise RuntimeError("ffmpeg stdin non disponible")

        perf = PerfCounter()
        perf.start()
        written = 0
        total_frames = None
        progress_width = 28
        last_progress = 0.0

        while True:
            item = frames_in.get()
            if item is stop_token:
                frames_in.task_done()
                break

            batch: FrameBatch = item
            if total_frames is None:
                total_frames = batch.total_frames
            t_batch0 = time.perf_counter()
            for frame in batch.frames:
                proc.stdin.write(frame.tobytes())
                written += 1
                perf.tick(1)

                if cfg.verbose and total_frames:
                    now = time.perf_counter()
                    if now - last_progress >= 0.25 or written == total_frames:
                        elapsed = now - perf.t0
                        pct = min(written / max(total_frames, 1), 1.0)
                        filled = int(progress_width * pct)
                        bar = "#" * filled + "-" * (progress_width - filled)
                        avg_fps = perf.frames / max(elapsed, 1e-6)
                        stats = (
                            f"{pct*100:5.1f}% | frames {written}/{total_frames}"
                            f" | avg {avg_fps:5.1f} fps | RAM ≈ {ram_mb():.0f} MB"
                        )
                        print(f"\r🚀 Rendering |{bar}| {stats}", end="", flush=True)
                        last_progress = now

            frames_in.task_done()
            if cfg.verbose and (
                batch.start_frame == 0 or batch.start_frame % (cfg.video.fps * 5) == 0
            ):
                dt_batch = time.perf_counter() - t_batch0
                if dt_batch > 0:
                    fps = len(batch.frames) / dt_batch
                    batch_bytes = sum(frame.nbytes for frame in batch.frames)
                    frame_mb = batch_memory_mb(batch.frames)
                    telemetry = format_batch_telemetry(
                        "📼 Encoder (consumer)",
                        batch.start_frame,
                        len(batch.frames),
                        batch_bytes,
                        frames_in,
                        fps,
                    )
                    print(f"{telemetry} | batch≈{frame_mb:.2f} MB (flush)")

        proc.stdin.close()
        return_code = proc.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg a échoué avec le code {return_code}")
        perf.stop()

        if cfg.verbose:
            if total_frames:
                pct = min(perf.frames / max(total_frames, 1), 1.0)
                bar = "#" * progress_width
                stats = (
                    f"100.0% | frames {perf.frames}/{total_frames}"
                    f" | avg {perf.avg_fps():5.1f} fps | RAM ≈ {ram_mb():.0f} MB"
                )
                print(f"\r🚀 Rendering |{bar}| {stats}")
            else:
                print("\n🚀 Rendering complete")
            print(f"✅ Vidéo ffmpeg terminée : {cfg.paths.out_video}")

        mux_audio(
            in_video=cfg.paths.out_video,
            audio_path=cfg.audio.audio_path,
            out_mp4=cfg.paths.out_final,
            verbose=cfg.verbose,
            verbose_lib=cfg.verbose_lib,
            encode=cfg.encode,
        )

    t = threading.Thread(target=_run, name="encoder_sink", daemon=True)
    t.start()
    return t
