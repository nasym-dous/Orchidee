import threading
from dataclasses import dataclass
from queue import Queue
from typing import List

import cv2
import numpy as np

from .config import AppConfig
from .types import AlphaBatch, AudioChunk


@dataclass
class _MeterValues:
    lufs: np.ndarray  # shape (2,)
    rms: np.ndarray   # shape (2,)
    true_peak: np.ndarray  # shape (2,)


def _dbfs(values: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(values + 1e-12)


def _db_to_level(db: float, floor: float = -30.0, ceiling: float = 0.0) -> float:
    return float(np.clip((db - floor) / (ceiling - floor), 0.0, 1.0))


def _compute_meter_series(samples: np.ndarray, spf: int) -> List[_MeterValues]:
    n_frames = max(int(np.ceil(samples.shape[0] / spf)), 1)
    result: List[_MeterValues] = []

    for frame_idx in range(n_frames):
        start = frame_idx * spf
        end = start + spf
        slice_lr = samples[start:end]
        if slice_lr.size == 0:
            slice_lr = np.zeros((1, 2), dtype=np.float32)

        rms = np.sqrt(np.mean(np.square(slice_lr), axis=0) + 1e-12)
        rms_db = _dbfs(rms)

        # simple approximation for LUFS based on RMS energy
        lufs_db = rms_db

        tp = np.max(np.abs(slice_lr), axis=0)
        tp_db = _dbfs(tp)

        result.append(_MeterValues(lufs=lufs_db, rms=rms_db, true_peak=tp_db))

    return result


def _draw_meter_channel(area: np.ndarray, values: _MeterValues, channel_idx: int):
    h, w = area.shape
    area[:] = 0

    levels = [
        (_db_to_level(values.true_peak[channel_idx]), 80),
        (_db_to_level(values.rms[channel_idx]), 160),
        (_db_to_level(values.lufs[channel_idx]), 240),
    ]

    for level, intensity in levels:
        height = int(level * (h - 4))
        if height <= 0:
            continue
        y0 = h - height
        cv2.rectangle(area, (1, y0), (w - 2, h - 2), int(intensity), thickness=-1)


def _draw_meters(alpha: np.ndarray, values: _MeterValues) -> np.ndarray:
    canvas = alpha.copy()
    h, w = canvas.shape
    meter_w = w//16

    # left-side blackout for meters
    canvas[:, :meter_w] = 0

    half_h = h // 2
    top = canvas[:half_h, :meter_w]
    bottom = canvas[half_h:, :meter_w]

    _draw_meter_channel(top, values, 0)
    _draw_meter_channel(bottom, values, 1)

    # info box in top-left corner
    box_h = min(72, h)
    box_w = min(int(w * 0.15), w)
    canvas[:box_h, meter_w:box_w+meter_w] = 0

    lines = [
        f"LUFS ({values.lufs[0]:2.1f}, {values.lufs[1]:2.1f})",
        f"RMS  ({values.rms[0]:2.1f}, {values.rms[1]:2.1f})",
        f"TP   ({values.true_peak[0]:2.1f}, {values.true_peak[1]:2.1f})",
    ]

    for i, text in enumerate(lines):
        y = 20 + i * 18
        cv2.putText(canvas, text, (6+meter_w, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 200, thickness=2, lineType=cv2.LINE_AA)

    return canvas


def start_telemetry_filter(
    cfg: AppConfig,
    audio_in: Queue,
    alpha_in: Queue,
    alpha_out: Queue,
    stop_token: object,
) -> threading.Thread:
    """Overlay stereo telemetry meters onto alpha batches."""

    def _run():
        item = audio_in.get()
        if item is stop_token:
            audio_in.task_done()
            alpha_out.put(stop_token)
            return

        chunk: AudioChunk = item
        audio_in.task_done()

        spf = max(int(cfg.audio.target_sr // cfg.video.fps), 1)
        meter_series = _compute_meter_series(chunk.samples, spf)

        while True:
            alpha_item = alpha_in.get()
            if alpha_item is stop_token:
                alpha_in.task_done()
                alpha_out.put(stop_token)
                break

            alpha_batch: AlphaBatch = alpha_item
            overlays = []
            for i, alpha in enumerate(alpha_batch.alphas):
                frame_idx = alpha_batch.start_frame + i
                meter_idx = min(frame_idx, len(meter_series) - 1)
                overlays.append(_draw_meters(alpha, meter_series[meter_idx]))

            alpha_out.put(AlphaBatch(start_frame=alpha_batch.start_frame, alphas=np.stack(overlays, axis=0)))
            alpha_in.task_done()

    t = threading.Thread(target=_run, name="telemetry_filter", daemon=True)
    t.start()
    return t

