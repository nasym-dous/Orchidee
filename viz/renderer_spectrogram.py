import numpy as np
from .config import AppConfig


def _prepare_window(n: int) -> np.ndarray:
    return np.hanning(n).astype(np.float32)


class SpectrogramRenderer:
    def __init__(self, audio_np: np.ndarray, cfg: AppConfig):
        self.cfg = cfg
        self.audio = audio_np.astype(np.float32)
        self.window_size = int(cfg.spectrogram.window_size)
        self.max_freq = float(cfg.spectrogram.max_freq_hz)
        self.scroll_px = max(int(cfg.spectrogram.scroll_px), 1)
        self.window = _prepare_window(self.window_size)

        self.spf = max(int(cfg.audio.target_sr // cfg.video.fps), 1)
        self.n_frames = int(np.ceil(self.audio.shape[0] / self.spf))

        self.freqs = np.fft.rfftfreq(self.window_size, d=1.0 / cfg.audio.target_sr)
        valid = self.freqs <= self.max_freq
        self.freqs = self.freqs[valid]
        self.valid_bins = valid
        if self.freqs.size == 0:
            raise ValueError("No frequency bins available below max_freq_hz")

        self.h = cfg.render.render_h
        self.w = cfg.render.render_w
        self.half_h = self.h // 2
        self.freq_axis = np.linspace(0.0, self.max_freq, self.half_h, dtype=np.float32)
        self.heat = np.zeros((self.h, self.w), dtype=np.float32)

        self.floor_db = float(cfg.spectrogram.floor_db)
        self.ceiling_db = 0.0

        if cfg.verbose:
            print("🎛 Spectrogram renderer")
            print(f"  frames         : {self.n_frames}")
            print(f"  window_size    : {self.window_size}")
            print(f"  scroll_px      : {self.scroll_px}")
            print(f"  max_freq_hz    : {self.max_freq}")

    def _slice_audio(self, start: int) -> np.ndarray:
        end = start + self.window_size
        segment = np.zeros((self.window_size, 2), dtype=np.float32)
        chunk = self.audio[start:end]
        segment[: chunk.shape[0]] = chunk
        return segment * self.window[:, None]

    def _slice_frame_audio(self, start: int) -> np.ndarray:
        end = start + self.spf
        segment = np.zeros((self.spf, 2), dtype=np.float32)
        chunk = self.audio[start:end]
        segment[: chunk.shape[0]] = chunk
        return segment

    def _compute_levels(self, start_sample: int) -> tuple[np.ndarray, np.ndarray]:
        frame_audio = self._slice_frame_audio(start_sample)
        power = np.mean(frame_audio ** 2, axis=0)
        rms_db = 10.0 * np.log10(np.maximum(power, 1e-12))

        lufs_offset = -0.691  # calibration constant from ITU-R BS.1770
        lufs_db = lufs_offset + 10.0 * np.log10(np.maximum(power, 1e-12))
        return rms_db.astype(np.float32), lufs_db.astype(np.float32)

    def _compute_columns(self, frame_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        start_sample = frame_idx * self.spf
        windowed = self._slice_audio(start_sample)
        rms_db, lufs_db = self._compute_levels(start_sample)

        spectrum = np.abs(np.fft.rfft(windowed, axis=0))[self.valid_bins]
        spectrum = np.maximum(spectrum, 1e-8)
        mag_db = 20.0 * np.log10(spectrum)
        norm = (mag_db - self.floor_db) / (self.ceiling_db - self.floor_db)
        norm = np.clip(norm, 0.0, 1.0)

        col_l = np.interp(self.freq_axis, self.freqs, norm[:, 0], left=0.0, right=0.0)
        col_r = np.interp(self.freq_axis, self.freqs, norm[:, 1], left=0.0, right=0.0)

        # invert so low frequencies are at the bottom
        col_l = col_l[::-1]
        col_r = col_r[::-1]

        col_l = np.tile(col_l[:, None] * self.cfg.scroll.gain, (1, self.scroll_px))
        col_r = np.tile(col_r[:, None] * self.cfg.scroll.gain, (1, self.scroll_px))
        return col_l.astype(np.float32), col_r.astype(np.float32), rms_db, lufs_db

    def _render_frame(self, frame_idx: int) -> np.ndarray:
        self.heat *= float(self.cfg.scroll.decay)
        self.heat[:, :-self.scroll_px] = self.heat[:, self.scroll_px:]
        self.heat[:, -self.scroll_px:] = 0.0

        col_l, col_r, rms_db, lufs_db = self._compute_columns(frame_idx)

        top = self.heat[: self.half_h]
        bottom = self.heat[self.half_h :]

        top[:, -self.scroll_px:] = np.maximum(top[:, -self.scroll_px:], col_l)
        bottom[:, -self.scroll_px:] = np.maximum(bottom[:, -self.scroll_px:], col_r)

        reveal_gain = float(self.cfg.scroll.reveal_gain)
        gamma = float(self.cfg.scroll.gamma)
        alpha = 1.0 - np.exp(-self.heat * reveal_gain)
        alpha = np.clip(alpha ** gamma, 0.0, 1.0)
        return (alpha * 255.0).astype(np.uint8), rms_db, lufs_db

    def next_alphas(self, t0: int, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        frames = []
        rms_levels = []
        lufs_levels = []
        for i in range(n):
            alpha, rms_db, lufs_db = self._render_frame(t0 + i)
            frames.append(alpha)
            rms_levels.append(rms_db)
            lufs_levels.append(lufs_db)
        return (
            np.stack(frames, axis=0),
            np.stack(rms_levels, axis=0),
            np.stack(lufs_levels, axis=0),
        )
