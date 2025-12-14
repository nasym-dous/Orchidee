import threading
from queue import Queue
import numpy as np
from .config import AppConfig
from .types import AlphaBatch


def _db_to_level(db_values: np.ndarray, floor_db: float) -> np.ndarray:
    norm = (db_values - floor_db) / abs(floor_db)
    return np.clip(norm, 0.0, 1.0)


def _draw_stereo_meter(alpha: np.ndarray, levels: np.ndarray, cfg: AppConfig, x0: int, floor_db: float) -> None:
    h, _ = alpha.shape
    pad = cfg.telemetry.padding_px
    bar_w = cfg.telemetry.bar_width_px
    gap = cfg.telemetry.channel_gap_px
    meter_h = int(h * cfg.telemetry.meter_height_ratio)

    y_bottom = h - pad
    y_top = max(pad, y_bottom - meter_h)

    norm_levels = _db_to_level(levels, floor_db)
    for ch in range(2):
        height = int(norm_levels[ch] * (y_bottom - y_top))
        bar_start_x = x0 + ch * (bar_w + gap)
        y_start = y_bottom - height
        alpha[y_start:y_bottom, bar_start_x : bar_start_x + bar_w] = np.maximum(
            alpha[y_start:y_bottom, bar_start_x : bar_start_x + bar_w], 255
        )


def _overlay_meters(alpha: np.ndarray, rms_db: np.ndarray, lufs_db: np.ndarray, cfg: AppConfig) -> np.ndarray:
    overlay = alpha.copy()
    x = cfg.telemetry.padding_px
    available_w = overlay.shape[1]

    meter_width = 2 * cfg.telemetry.bar_width_px + cfg.telemetry.channel_gap_px
    total_width = meter_width * 2 + cfg.telemetry.meter_gap_px + 2 * cfg.telemetry.padding_px
    if total_width > available_w:
        return overlay

    _draw_stereo_meter(overlay, lufs_db, cfg, x, cfg.telemetry.floor_lufs_db)
    x += 2 * cfg.telemetry.bar_width_px + cfg.telemetry.channel_gap_px + cfg.telemetry.meter_gap_px

    _draw_stereo_meter(overlay, rms_db, cfg, x, cfg.telemetry.floor_rms_db)
    return overlay


def start_telemetry_filter(cfg: AppConfig, alpha_in: Queue, alpha_out: Queue, stop_token: object) -> threading.Thread:
    """Overlay audio telemetry meters on top of the alpha batches."""

    def _run():
        while True:
            item = alpha_in.get()
            if item is stop_token:
                alpha_in.task_done()
                alpha_out.put(stop_token)
                break

            batch: AlphaBatch = item
            telemetry = batch.telemetry or {}
            rms_levels = telemetry.get("rms_db")
            lufs_levels = telemetry.get("lufs_db")

            if rms_levels is None or lufs_levels is None or not cfg.telemetry.enabled:
                alpha_out.put(batch)
                alpha_in.task_done()
                continue

            augmented = [
                _overlay_meters(alpha, rms, lufs, cfg)
                for alpha, rms, lufs in zip(batch.alphas, rms_levels, lufs_levels)
            ]

            alpha_out.put(
                AlphaBatch(
                    start_frame=batch.start_frame,
                    alphas=np.stack(augmented, axis=0),
                    telemetry=batch.telemetry,
                )
            )
            alpha_in.task_done()

    t = threading.Thread(target=_run, name="telemetry_filter", daemon=True)
    t.start()
    return t
