import numpy as np

from .config import AppConfig


class SpectrogramRenderer:
    def __init__(self, audio_np: np.ndarray, cfg: AppConfig):
        self.cfg = cfg
        self.audio_mono = audio_np.mean(axis=1).astype(np.float32)
        self.spf = max(int(cfg.audio.target_sr // cfg.video.fps), 1)
        self.n_frames = int(np.ceil(self.audio_mono.shape[0] / self.spf))

        # FFT sizing: small enough to be responsive but detailed enough for the spectrum.
        self.n_fft = max(512, min(self.spf * 4, 4096))
        self.window = np.hanning(self.n_fft).astype(np.float32)

        self.buffer = np.zeros((cfg.render.render_h, cfg.render.render_w), dtype=np.float32)
        self.freq_axis = None

    def _column_for_frame(self, frame_idx: int) -> np.ndarray:
        start = frame_idx * self.spf
        end = start + self.n_fft

        segment = np.zeros(self.n_fft, dtype=np.float32)
        if start < self.audio_mono.shape[0]:
            take = min(self.audio_mono.shape[0] - start, self.n_fft)
            segment[:take] = self.audio_mono[start:start + take]

        spectrum = np.fft.rfft(segment * self.window)
        magnitude = np.log1p(np.abs(spectrum)).astype(np.float32)

        # Lazily build interpolation axes for the spectrogram column.
        if self.freq_axis is None:
            self.freq_axis = np.linspace(0, magnitude.size - 1, magnitude.size)
            self.target_freq = np.linspace(0, magnitude.size - 1, self.cfg.render.render_h)

        column = np.interp(self.target_freq, self.freq_axis, magnitude).astype(np.float32)
        column -= column.min()
        if column.max() > 0:
            column /= column.max()
        return column

    def next_alphas(self, t0: int, n: int) -> np.ndarray:
        frames: list[np.ndarray] = []
        for i in range(n):
            frame_idx = t0 + i
            if frame_idx >= self.n_frames:
                column = np.zeros(self.cfg.render.render_h, dtype=np.float32)
            else:
                column = self._column_for_frame(frame_idx)

            self.buffer = np.roll(self.buffer, -1, axis=1)
            self.buffer[:, -1] = column

            scale = self.buffer.max()
            alpha = self.buffer / scale if scale > 0 else self.buffer
            frames.append((alpha * 255.0).astype(np.uint8))

        return np.stack(frames, axis=0)
