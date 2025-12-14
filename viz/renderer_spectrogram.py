import numpy as np

from .config import AppConfig


def _stft_mag(audio: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    window = np.hanning(n_fft).astype(np.float32)
    frames = []
    for start in range(0, max(len(audio) - n_fft, 1), hop):
        end = start + n_fft
        chunk = audio[start:end]
        if chunk.shape[0] < n_fft:
            pad = np.zeros(n_fft - chunk.shape[0], dtype=chunk.dtype)
            chunk = np.concatenate([chunk, pad])
        chunk = chunk * window
        spec = np.fft.rfft(chunk)
        frames.append(np.abs(spec))
    if not frames:
        return np.zeros((0, n_fft // 2 + 1), dtype=np.float32)
    return np.stack(frames, axis=0)


def _db_norm(mag: np.ndarray, dynamic_range: float, gamma: float, gain: float) -> np.ndarray:
    mag = mag * gain
    mag = np.maximum(mag, 1e-8)
    db = 20.0 * np.log10(mag)
    db_max = np.max(db)
    db_min = db_max - dynamic_range
    scaled = (db - db_min) / max(dynamic_range, 1e-6)
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled ** gamma) * 255.0


def _resize_freq(db_frames: np.ndarray, sr: int, target_h: int) -> np.ndarray:
    if db_frames.shape[0] == 0:
        return np.zeros((0, target_h), dtype=np.float32)

    freqs = np.linspace(0.0, sr * 0.5, db_frames.shape[1], dtype=np.float32)
    target_freqs = np.linspace(0.0, sr * 0.5, target_h, dtype=np.float32)

    resized = np.zeros((db_frames.shape[0], target_h), dtype=np.float32)
    for i, frame in enumerate(db_frames):
        resized[i] = np.interp(target_freqs, freqs, frame)
    return resized


class SpectrogramRenderer:
    def __init__(self, audio_np: np.ndarray, cfg: AppConfig):
        self.cfg = cfg
        self.w = cfg.render.render_w
        self.h = cfg.render.render_h

        audio_mono = audio_np.mean(axis=1).astype(np.float32)
        hop = max(int(cfg.audio.target_sr // cfg.video.fps), 1)

        mag = _stft_mag(audio_mono, cfg.spectrogram.n_fft, hop)
        db = _db_norm(mag, cfg.spectrogram.dynamic_range, cfg.spectrogram.gamma, cfg.spectrogram.gain)
        db_resized = _resize_freq(db, cfg.audio.target_sr, self.h)

        spectrogram = np.flipud(db_resized.T)  # (H, time)
        time_bins = spectrogram.shape[1]

        self.frames = []
        for t in range(time_bins):
            start = max(0, t - self.w + 1)
            window = spectrogram[:, start:t + 1]
            if window.shape[1] < self.w:
                pad = np.zeros((self.h, self.w - window.shape[1]), dtype=np.float32)
                window = np.concatenate([pad, window], axis=1)
            self.frames.append(window.astype(np.uint8))

        self.n_frames = len(self.frames)

    def next_alphas(self, t0: int, n: int) -> np.ndarray:
        t1 = min(t0 + n, self.n_frames)
        return np.stack(self.frames[t0:t1], axis=0)
