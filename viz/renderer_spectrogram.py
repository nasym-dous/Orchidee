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
        self.min_freq = float(cfg.spectrogram.min_hz_bound)
        self.max_freq = float(cfg.spectrogram.max_freq_hz)
        self.scroll_px = max(int(cfg.spectrogram.scroll_px), 1)
        self.write_px = min(max(int(cfg.spectrogram.write_px), 1), self.scroll_px)
        self.tilt_db_per_octave = float(cfg.spectrogram.tilt_db_per_octave)
        self.window = _prepare_window(self.window_size)
        self.windowed_buf = np.zeros((self.window_size, 2), dtype=np.float32)
        self.segment_buf = np.zeros((self.window_size, 2), dtype=np.float32)

        self.spf = max(int(cfg.audio.target_sr // cfg.video.fps), 1)
        self.n_frames = int(np.ceil(self.audio.shape[0] / self.spf))

        self.freqs = np.fft.rfftfreq(self.fft_size, d=1.0 / cfg.audio.target_sr)
        valid = (self.freqs >= self.min_freq) & (self.freqs <= self.max_freq)
        self.freqs = self.freqs[valid]
        self.valid_bins = valid
        if self.freqs.size == 0:
            raise ValueError(
                f"No frequency bins available between {self.min_freq}Hz and {self.max_freq}Hz"
            )

        positive_freqs = self.freqs[self.freqs > 0]
        if positive_freqs.size == 0:
            raise ValueError("No positive frequency bins available for log scaling")

        # Replace the DC bin with the smallest positive bin so log scaling works
        if self.freqs[0] == 0.0:
            self.freqs = self.freqs.copy()
            self.freqs[0] = float(positive_freqs.min())

        self.min_freq = float(max(self.min_freq, positive_freqs.min()))
        self.log_freqs = np.log10(self.freqs)

        self.h = cfg.render.render_h
        self.w = cfg.render.render_w
        self.half_h = self.h // 2
        self.freq_axis = np.logspace(
            np.log10(self.min_freq), np.log10(self.max_freq), self.half_h, dtype=np.float32
        )
        # Build log-spaced band edges so we can pool energy into each rendered row
        # instead of linearly interpolating. This keeps low-frequency details sharp
        # where bins are sparse.
        freq_edges = np.empty(self.half_h + 1, dtype=np.float32)
        freq_edges[0] = self.min_freq
        freq_edges[-1] = self.max_freq
        freq_edges[1:-1] = np.sqrt(self.freq_axis[:-1] * self.freq_axis[1:])
        self.band_bin_edges = np.clip(
            np.searchsorted(self.freqs, freq_edges, side="left"), 0, self.freqs.size
        )
        octaves_from_min = np.log2(self.freq_axis / self.min_freq)
        self.freq_tilt_gain = (10.0 ** ((octaves_from_min * self.tilt_db_per_octave) / 20.0)).astype(
            np.float32
        )
        self.heat = np.zeros((self.h, self.w), dtype=np.float32)

        fft_bins = np.count_nonzero(self.valid_bins)
        self.spectrum_buf = np.zeros((fft_bins, 2), dtype=np.float32)
        self.norm_buf = np.zeros_like(self.spectrum_buf)

        if cfg.verbose:
            print("🎛 Spectrogram renderer")
            print(f"  frames         : {self.n_frames}")
            print(f"  window_size    : {self.window_size}")
            print(f"  scroll_px      : {self.scroll_px}")
            print(f"  write_px       : {self.write_px}")
            print(f"  fft_size       : {self.fft_size}")
            print(f"  min_freq_hz    : {self.min_freq}")
            print(f"  max_freq_hz    : {self.max_freq}")
            if self.tilt_db_per_octave != 0.0:
                print(f"  tilt_db/octave : {self.tilt_db_per_octave}")

    def _slice_audio(self, start: int) -> np.ndarray:
        end = start + self.window_size
        segment = self.segment_buf
        segment.fill(0.0)
        chunk = self.audio[start:end]
        segment[: chunk.shape[0]] = chunk
        np.multiply(segment, self.window[:, None], out=self.windowed_buf)
        return self.windowed_buf

    def _compute_column_at(self, start_sample: int) -> tuple[np.ndarray, np.ndarray]:
        windowed = self._slice_audio(start_sample)

        spectrum = np.fft.rfft(windowed, n=self.fft_size, axis=0)[self.valid_bins]
        np.abs(spectrum, out=self.spectrum_buf)
        np.maximum(self.spectrum_buf, 1e-12, out=self.spectrum_buf)

        peak = float(np.max(self.spectrum_buf))
        if peak <= 0.0:
            peak = 1.0

        np.divide(self.spectrum_buf, peak, out=self.norm_buf)
        np.clip(self.norm_buf, 0.0, 1.0, out=self.norm_buf)

        col_l = np.empty(self.half_h, dtype=np.float32)
        col_r = np.empty_like(col_l)

        for i in range(self.half_h):
            start = int(self.band_bin_edges[i])
            end = int(self.band_bin_edges[i + 1])

            if start >= self.norm_buf.shape[0]:
                start = self.norm_buf.shape[0] - 1

            if start >= end:
                val_l = self.norm_buf[start, 0]
                val_r = self.norm_buf[start, 1]
            else:
                val_l = float(np.max(self.norm_buf[start:end, 0]))
                val_r = float(np.max(self.norm_buf[start:end, 1]))

            col_l[i] = val_l
            col_r[i] = val_r

        if self.tilt_db_per_octave != 0.0:
            col_l *= self.freq_tilt_gain
            col_r *= self.freq_tilt_gain

        # invert so low frequencies are at the bottom
        col_l = col_l[::-1]
        col_r = col_r[::-1]

        return col_l.astype(np.float32), col_r.astype(np.float32)

    def _compute_columns(self, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        start_sample = frame_idx * self.spf
        columns_to_generate = int(np.ceil(self.scroll_px / self.write_px))
        hop = self.spf / max(columns_to_generate, 1)

        cols_l: list[np.ndarray] = []
        cols_r: list[np.ndarray] = []

        for i in range(columns_to_generate):
            col_l, col_r = self._compute_column_at(int(start_sample + i * hop))
            col_l = np.tile(col_l[:, None], (1, self.write_px))
            col_r = np.tile(col_r[:, None], (1, self.write_px))
            cols_l.append(col_l)
            cols_r.append(col_r)

        col_l_full = np.concatenate(cols_l, axis=1) * self.cfg.scroll.gain
        col_r_full = np.concatenate(cols_r, axis=1) * self.cfg.scroll.gain

        if col_l_full.shape[1] > self.scroll_px:
            col_l_full = col_l_full[:, -self.scroll_px :]
            col_r_full = col_r_full[:, -self.scroll_px :]

        return col_l_full.astype(np.float32), col_r_full.astype(np.float32)

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
