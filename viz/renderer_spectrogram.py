import cv2
import numpy as np

from .config import AppConfig


class SpectrogramRenderer:
    def __init__(self, audio_np: np.ndarray, cfg: AppConfig):
        self.cfg = cfg
        self.render_w = cfg.render.render_w
        self.render_h = cfg.render.render_h

        mono = audio_np.mean(axis=1)

        self.hop_length = max(int(cfg.audio.target_sr // cfg.video.fps), 1)
        self.n_fft = 1024

        self.spec = self._compute_spectrogram(mono)
        self.spec_img = self._normalize(self.spec)

        self.n_frames = self.spec_img.shape[0]
        self.n_bins = self.spec_img.shape[1]

    def _compute_spectrogram(self, mono: np.ndarray) -> np.ndarray:
        window = np.hanning(self.n_fft)

        if mono.size < self.n_fft:
            pad_width = self.n_fft - mono.size
            mono = np.pad(mono, (0, pad_width), mode="constant")

        n_frames = max(1, 1 + (mono.size - self.n_fft) // self.hop_length)
        spec = np.zeros((n_frames, self.n_fft // 2 + 1), dtype=np.float32)

        for i in range(n_frames):
            start = i * self.hop_length
            end = start + self.n_fft
            frame = mono[start:end]
            if frame.size < self.n_fft:
                frame = np.pad(frame, (0, self.n_fft - frame.size), mode="constant")

            spectrum = np.abs(np.fft.rfft(frame * window))
            spec[i] = spectrum

        return spec

    def _normalize(self, spec: np.ndarray) -> np.ndarray:
        spec_db = np.log1p(spec)
        spec_db -= spec_db.min()
        max_val = spec_db.max()

        if max_val > 0:
            spec_db /= max_val

        return (spec_db * 255.0).astype(np.uint8)

    def _alpha_for_frame(self, frame_idx: int) -> np.ndarray:
        start = max(0, frame_idx - self.render_w + 1)
        end = min(frame_idx + 1, self.n_frames)

        slice_img = self.spec_img[start:end]
        canvas = np.zeros((self.render_w, self.n_bins), dtype=np.uint8)
        canvas[-slice_img.shape[0]:, :] = slice_img

        freq_time = canvas.T
        alpha = cv2.resize(freq_time, (self.render_w, self.render_h), interpolation=cv2.INTER_LINEAR)
        return alpha

    def next_alphas(self, t0: int, n: int) -> np.ndarray:
        alphas = [self._alpha_for_frame(t0 + i) for i in range(n) if (t0 + i) < self.n_frames]
        return np.asarray(alphas)
