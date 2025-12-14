import threading
from dataclasses import dataclass
from queue import Queue
from typing import List, Tuple

import numpy as np

from .config import AppConfig
from .types import AlphaBatch, AudioChunk


def _prepare_window(n: int) -> np.ndarray:
    return np.hanning(n).astype(np.float32)


@dataclass
class _TracePoint:
    root: float | None
    formants: List[float]


class _RootFormantAnalyzer:
    def __init__(self, samples: np.ndarray, cfg: AppConfig):
        self.samples = samples.astype(np.float32)
        self.cfg = cfg
        self.window_size = int(cfg.spectrogram.window_size)
        self.window = _prepare_window(self.window_size)
        self.max_freq = float(cfg.spectrogram.max_freq_hz)
        self.scroll_px = max(int(cfg.spectrogram.scroll_px), 1)
        self.half_h = cfg.render.render_h // 2

        self.spf = max(int(cfg.audio.target_sr // cfg.video.fps), 1)
        self.n_frames = int(np.ceil(self.samples.shape[0] / self.spf))

        full_freqs = np.fft.rfftfreq(self.window_size, d=1.0 / cfg.audio.target_sr)
        valid = full_freqs <= self.max_freq
        self.freqs = full_freqs[valid]
        self.valid_bins = valid

        self.points: List[Tuple[_TracePoint, _TracePoint]] = []
        self._compute_traces()

    def _slice_audio(self, start: int) -> np.ndarray:
        end = start + self.window_size
        segment = np.zeros((self.window_size, 2), dtype=np.float32)
        chunk = self.samples[start:end]
        segment[: chunk.shape[0]] = chunk
        return segment * self.window[:, None]

    def _analyze_channel(self, spectrum: np.ndarray) -> _TracePoint:
        cfg = self.cfg.formants
        mags = np.maximum(spectrum, 1e-8)

        mask = (self.freqs >= cfg.root_min_hz) & (self.freqs <= cfg.root_max_hz)
        if not np.any(mask):
            return _TracePoint(root=None, formants=[])

        masked_mag = mags.copy()
        masked_mag[~mask] = 0.0
        root_idx = int(np.argmax(masked_mag))
        root_freq = float(self.freqs[root_idx])

        formant_mag = mags.copy()
        neighborhood = np.abs(self.freqs - root_freq) <= cfg.neighborhood_hz
        formant_mag[neighborhood] = 0.0

        n_formants = cfg.num_formants
        if n_formants == 0:
            return _TracePoint(root=root_freq, formants=[])

        if formant_mag.size == 0:
            return _TracePoint(root=root_freq, formants=[])

        candidate_indices = np.argpartition(formant_mag, -n_formants)[-n_formants:]
        candidate_indices = candidate_indices[np.argsort(formant_mag[candidate_indices])[::-1]]

        formants: List[float] = []
        for idx in candidate_indices:
            if formant_mag[idx] <= 0.0:
                continue
            formants.append(float(self.freqs[idx]))

        return _TracePoint(root=root_freq, formants=formants)

    def _compute_traces(self):
        for frame_idx in range(self.n_frames):
            start = frame_idx * self.spf
            windowed = self._slice_audio(start)
            spectrum = np.abs(np.fft.rfft(windowed, axis=0))[self.valid_bins]

            left = self._analyze_channel(spectrum[:, 0])
            right = self._analyze_channel(spectrum[:, 1])
            self.points.append((left, right))

    def _freq_to_row(self, freq: float) -> int:
        pos = 1.0 - (freq / self.max_freq)
        row = int(np.clip(pos * self.half_h, 0, self.half_h - 1))
        return row

    def draw_frame(self, mask: np.ndarray, frame_idx: int):
        if frame_idx >= len(self.points):
            return

        left, right = self.points[frame_idx]
        thickness = max(int(self.cfg.formants.trace_thickness), 1)
        half_h = self.half_h
        scroll_px = self.scroll_px

        def _draw(channel_point: _TracePoint, y_offset: int):
            if channel_point.root is not None:
                row = y_offset + self._freq_to_row(channel_point.root)
                row_end = min(row + thickness, mask.shape[0])
                mask[row:row_end, -scroll_px:] = True
            for formant in channel_point.formants:
                row = y_offset + self._freq_to_row(formant)
                row_end = min(row + thickness, mask.shape[0])
                mask[row:row_end, -scroll_px:] = True

        _draw(left, 0)
        _draw(right, half_h)


def start_formant_trace_filter(
    cfg: AppConfig,
    audio_in: Queue,
    alpha_in: Queue,
    alpha_out: Queue,
    stop_token: object,
) -> threading.Thread:
    """Overlay root and formant traces onto spectrogram alpha batches."""

    def _run():
        item = audio_in.get()
        if item is stop_token:
            audio_in.task_done()
            alpha_out.put(stop_token)
            return

        chunk: AudioChunk = item
        audio_in.task_done()

        if not cfg.formants.enabled:
            while True:
                alpha_item = alpha_in.get()
                if alpha_item is stop_token:
                    alpha_in.task_done()
                    alpha_out.put(stop_token)
                    break
                alpha_out.put(alpha_item)
                alpha_in.task_done()
            return

        analyzer = _RootFormantAnalyzer(chunk.samples, cfg)

        trace_mask = np.zeros((cfg.render.render_h, cfg.render.render_w), dtype=bool)

        while True:
            alpha_item = alpha_in.get()
            if alpha_item is stop_token:
                alpha_in.task_done()
                alpha_out.put(stop_token)
                break

            alpha_batch: AlphaBatch = alpha_item
            overlays = []

            for i, alpha in enumerate(alpha_batch.alphas):
                # scroll the trace mask to match the spectrogram movement
                trace_mask[:, :-analyzer.scroll_px] = trace_mask[:, analyzer.scroll_px:]
                trace_mask[:, -analyzer.scroll_px:] = False

                frame_idx = alpha_batch.start_frame + i
                analyzer.draw_frame(trace_mask, frame_idx)

                stamped = alpha.copy()
                stamped[trace_mask] = 0
                overlays.append(stamped)

            alpha_out.put(AlphaBatch(start_frame=alpha_batch.start_frame, alphas=np.stack(overlays, axis=0)))
            alpha_in.task_done()

    t = threading.Thread(target=_run, name="formant_trace_filter", daemon=True)
    t.start()
    return t

