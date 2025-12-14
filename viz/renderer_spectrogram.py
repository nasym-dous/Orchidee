import numpy as np
from .config import AppConfig


def _prepare_window(n: int) -> np.ndarray:
    return np.hanning(n).astype(np.float32)


class SpectrogramRenderer:
    def __init__(self, audio_np: np.ndarray, cfg: AppConfig):
        self.cfg = cfg
        self.audio = audio_np.astype(np.float32)
        self.window_size = int(cfg.spectrogram.window_size)
        self.fft_size = int(cfg.spectrogram.fft_size)
        self.max_freq = float(cfg.spectrogram.max_freq_hz)
        self.scroll_px = max(int(cfg.spectrogram.scroll_px), 1)
        self.window = _prepare_window(self.window_size)
        self.windowed_buf = np.zeros((self.window_size, 2), dtype=np.float32)
        self.segment_buf = np.zeros((self.window_size, 2), dtype=np.float32)

        self.spf = max(int(cfg.audio.target_sr // cfg.video.fps), 1)
        self.n_frames = int(np.ceil(self.audio.shape[0] / self.spf))

        self.freqs = np.fft.rfftfreq(self.fft_size, d=1.0 / cfg.audio.target_sr)
        valid = self.freqs <= self.max_freq
        self.freqs = self.freqs[valid]
        self.valid_bins = valid
        if self.freqs.size == 0:
            raise ValueError("No frequency bins available below max_freq_hz")

        positive_freqs = self.freqs[self.freqs > 0]
        if positive_freqs.size == 0:
            raise ValueError("No positive frequency bins available for log scaling")

        # Replace the DC bin with the smallest positive bin so log scaling works
        if self.freqs[0] == 0.0:
            self.freqs = self.freqs.copy()
            self.freqs[0] = float(positive_freqs.min())

        self.min_freq = float(positive_freqs.min())
        self.log_freqs = np.log10(self.freqs)

        self.h = cfg.render.render_h
        self.w = cfg.render.render_w
        self.half_h = self.h // 2
        self.freq_axis = np.logspace(
            np.log10(self.min_freq), np.log10(self.max_freq), self.half_h, dtype=np.float32
        )
        self.log_freq_axis = np.log10(self.freq_axis)
        self.heat = np.zeros((self.h, self.w), dtype=np.float32)

        self.floor_db = float(cfg.spectrogram.floor_db)
        self.ceiling_db = float(cfg.spectrogram.ceiling_db)
        self.norm_denom = max(self.ceiling_db - self.floor_db, 1e-6)

        fft_bins = np.count_nonzero(self.valid_bins)
        self.spectrum_buf = np.zeros((fft_bins, 2), dtype=np.float32)
        self.mag_db_buf = np.zeros_like(self.spectrum_buf)
        self.norm_buf = np.zeros_like(self.spectrum_buf)

        if cfg.verbose:
            print("🎛 Spectrogram renderer")
            print(f"  frames         : {self.n_frames}")
            print(f"  window_size    : {self.window_size}")
            print(f"  scroll_px      : {self.scroll_px}")
            print(f"  fft_size       : {self.fft_size}")
            print(f"  max_freq_hz    : {self.max_freq}")

    def _slice_audio(self, start: int) -> np.ndarray:
        end = start + self.window_size
        segment = self.segment_buf
        segment.fill(0.0)
        chunk = self.audio[start:end]
        segment[: chunk.shape[0]] = chunk
        np.multiply(segment, self.window[:, None], out=self.windowed_buf)
        return self.windowed_buf

    def _compute_columns(self, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        start_sample = frame_idx * self.spf
        windowed = self._slice_audio(start_sample)

        spectrum = np.fft.rfft(windowed, n=self.fft_size, axis=0)[self.valid_bins]
        np.abs(spectrum, out=self.spectrum_buf)
        np.maximum(self.spectrum_buf, 1e-12, out=self.spectrum_buf)

        np.log10(self.spectrum_buf, out=self.mag_db_buf)
        self.mag_db_buf *= 20.0

        np.subtract(self.mag_db_buf, self.floor_db, out=self.norm_buf)
        self.norm_buf *= 1.0 / self.norm_denom
        np.clip(self.norm_buf, 0.0, 1.0, out=self.norm_buf)

        col_l = np.interp(
            self.log_freq_axis, self.log_freqs, self.norm_buf[:, 0], left=0.0, right=0.0
        )
        col_r = np.interp(
            self.log_freq_axis, self.log_freqs, self.norm_buf[:, 1], left=0.0, right=0.0
        )

        # invert so low frequencies are at the bottom
        col_l = col_l[::-1]
        col_r = col_r[::-1]

        col_l = np.tile(col_l[:, None] * self.cfg.scroll.gain, (1, self.scroll_px))
        col_r = np.tile(col_r[:, None] * self.cfg.scroll.gain, (1, self.scroll_px))
        return col_l.astype(np.float32), col_r.astype(np.float32)

    def _render_frame(self, frame_idx: int) -> np.ndarray:
        self.heat *= float(self.cfg.scroll.decay)
        self.heat[:, :-self.scroll_px] = self.heat[:, self.scroll_px:]
        self.heat[:, -self.scroll_px:] = 0.0

        col_l, col_r = self._compute_columns(frame_idx)

        top = self.heat[: self.half_h]
        bottom = self.heat[self.half_h :]

        top[:, -self.scroll_px:] = np.maximum(top[:, -self.scroll_px:], col_l)
        bottom[:, -self.scroll_px:] = np.maximum(bottom[:, -self.scroll_px:], col_r)

        reveal_gain = float(self.cfg.scroll.reveal_gain)
        gamma = float(self.cfg.scroll.gamma)
        alpha = 1.0 - np.exp(-self.heat * reveal_gain)
        alpha = np.clip(alpha ** gamma, 0.0, 1.0)
        return (alpha * 255.0).astype(np.uint8)

    def next_alphas(self, t0: int, n: int) -> np.ndarray:
        frames = [self._render_frame(t0 + i) for i in range(n)]
        return np.stack(frames, axis=0)
