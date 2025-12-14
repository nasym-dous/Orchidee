import numpy as np


class SpectrogramRenderer:
    """Simple spectrogram-style renderer.

    Transforms audio into a scrolling time-frequency visualization where each
    new frame contributes a column on the right side of the image.
    """

    def __init__(self, audio_np: np.ndarray, sr: int, render_w: int, render_h: int, fps: int):
        self.audio = self._to_mono(audio_np)
        self.sample_rate = sr
        self.render_w = render_w
        self.render_h = render_h
        self.fps = fps

        self.spf = max(int(self.sample_rate // self.fps), 1)
        self.n_frames = int(np.ceil(self.audio.shape[0] / self.spf))

        self.window_size = min(2048, self.audio.shape[0])
        self.window = np.hanning(self.window_size).astype(np.float32)

        self.spectrogram = np.zeros((self.render_h, self.render_w), dtype=np.uint8)

    @staticmethod
    def _to_mono(audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 1:
            return audio.astype(np.float32)
        return audio.astype(np.float32).mean(axis=1)

    def _compute_column(self, start_sample: int) -> np.ndarray:
        window = self.audio[start_sample:start_sample + self.window_size]
        if window.shape[0] < self.window_size:
            window = np.pad(window, (0, self.window_size - window.shape[0]))

        windowed = window * self.window
        spectrum = np.abs(np.fft.rfft(windowed))

        magnitude_db = 20.0 * np.log10(spectrum + 1e-6)
        magnitude_db -= magnitude_db.max()
        min_db = -60.0
        magnitude_db = np.clip(magnitude_db, min_db, 0.0)

        norm = (magnitude_db - min_db) / -min_db
        freq_axis = np.linspace(0, norm.shape[0] - 1, self.render_h).astype(int)
        column = norm[freq_axis]
        return (column * 255.0).astype(np.uint8)

    def next_alphas(self, t0: int, n: int) -> np.ndarray:
        frames = []
        for i in range(n):
            start = (t0 + i) * self.spf
            column = self._compute_column(start)

            self.spectrogram[:, :-1] = self.spectrogram[:, 1:]
            self.spectrogram[:, -1] = column

            frames.append(self.spectrogram.copy())

        return np.stack(frames, axis=0)
