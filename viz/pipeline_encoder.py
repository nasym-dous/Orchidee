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

    def _writer_params() -> list:
        """Build OpenCV VideoWriter params for optional hardware acceleration.

        If the current OpenCV build includes VideoToolbox (or other accelerators),
        setting the VIDEOWRITER_PROP_HW_ACCELERATION property is enough to enable
        it. Otherwise, the user needs an OpenCV build compiled with that backend.
        """

        if cfg.video.hw_accel is None:
            return []

        prop_hw_accel = getattr(cv2, "VIDEOWRITER_PROP_HW_ACCELERATION", None)
        if prop_hw_accel is None:
            raise RuntimeError(
                "OpenCV does not expose VIDEOWRITER_PROP_HW_ACCELERATION; "
                "install a build compiled with the desired hardware codec (e.g. VideoToolbox)."
            )

        accel_map = {
            "videotoolbox": getattr(cv2, "VIDEO_ACCELERATION_VIDEOTOOLBOX", None),
            "vaapi": getattr(cv2, "VIDEO_ACCELERATION_VAAPI", None),
            "any": getattr(cv2, "VIDEO_ACCELERATION_ANY", None),
            "none": getattr(cv2, "VIDEO_ACCELERATION_NONE", None),
        }

        accel_value = accel_map.get(cfg.video.hw_accel)
        if accel_value is None:
            raise RuntimeError(
                f"Hardware acceleration '{cfg.video.hw_accel}' is not available in this OpenCV build."
            )

        return [prop_hw_accel, accel_value]

    def _run():
        out = cv2.VideoWriter(
            cfg.paths.out_avi,
            cv2.VideoWriter_fourcc(*cfg.video.fourcc),
            cfg.video.fps,
            (cfg.video.w, cfg.video.h),
            params=_writer_params(),
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
            encode=cfg.encode,
        )

    t = threading.Thread(target=_run, name="encoder_sink", daemon=True)
    t.start()
    return t
