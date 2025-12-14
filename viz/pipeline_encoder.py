import threading
import time
from queue import Queue
import cv2
from .config import AppConfig
from .encode import mux_audio
from .stats import PerfCounter, batch_memory_mb, format_batch_telemetry, ram_mb
from .types import FrameBatch


def start_encoder_sink(cfg: AppConfig, frames_in: Queue, stop_token: object) -> threading.Thread:
    """Encode frames to disk and mux the original audio."""

    def _run():
        out = cv2.VideoWriter(
            cfg.paths.out_avi,
            cv2.VideoWriter_fourcc(*cfg.video.fourcc),
            cfg.video.fps,
            (cfg.video.w, cfg.video.h),
        )
        if not out.isOpened():
            raise RuntimeError("VideoWriter non ouvert")

        perf = PerfCounter()
        perf.start()
        written = 0

        while True:
            item = frames_in.get()
            if item is stop_token:
                frames_in.task_done()
                break

            batch: FrameBatch = item
            t_batch0 = time.perf_counter()
            for frame in batch.frames:
                out.write(frame)
                written += 1
                perf.tick(1)

                if cfg.verbose and written % (cfg.video.fps * 5) == 0:
                    avg_fps = perf.frames / max(time.perf_counter() - perf.t0, 1e-6)
                    batch_bytes = sum(frame.nbytes for frame in batch.frames)
                    frame_mb = batch_memory_mb(batch.frames)
                    telemetry = format_batch_telemetry(
                        "📼 Encoder (consumer)",
                        batch.start_frame,
                        len(batch.frames),
                        batch_bytes,
                        frames_in,
                        avg_fps,
                    )
                    print(f"{telemetry} | batch≈{frame_mb:.2f} MB | RAM ≈ {ram_mb():.0f} MB")

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

        out.release()
        perf.stop()

        if cfg.verbose:
            print("🚀 Render performance")
            print(f"  frames rendered : {perf.frames}")
            print(f"  avg FPS         : {perf.avg_fps():.2f}")
            print(f"  RAM (now)       : {ram_mb():.0f} MB")
            print(f"✅ Vidéo MJPG terminée : {cfg.paths.out_avi}")

        mux_audio(
            in_avi=cfg.paths.out_avi,
            audio_path=cfg.audio.audio_path,
            out_mp4=cfg.paths.out_final,
            verbose=cfg.verbose,
        )

    t = threading.Thread(target=_run, name="encoder_sink", daemon=True)
    t.start()
    return t
