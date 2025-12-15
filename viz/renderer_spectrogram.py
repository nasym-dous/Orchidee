import numpy as np
import jax
import jax.numpy as jnp
from .config import AppConfig


def _prepare_window(n: int) -> np.ndarray:
    """Return a Hann window of length ``n`` as ``float32``."""

    return np.hanning(n).astype(np.float32)


def _compute_frequency_axis(cfg: AppConfig, fft_size: int, half_h: int):
    """Prepare logarithmic frequency metadata for the spectrogram display."""

    min_freq = float(cfg.spectrogram.min_hz_bound)
    max_freq = float(cfg.spectrogram.max_freq_hz)

    freqs = np.fft.rfftfreq(fft_size, d=1.0 / cfg.audio.target_sr)
    valid_bins = (freqs >= min_freq) & (freqs <= max_freq)
    freqs = freqs[valid_bins]
    valid_indices = np.nonzero(valid_bins)[0].astype(np.int32)
    if freqs.size == 0:
        raise ValueError(
            f"No frequency bins available between {min_freq}Hz and {max_freq}Hz"
        )

    positive_freqs = freqs[freqs > 0]
    if positive_freqs.size == 0:
        raise ValueError("No positive frequency bins available for log scaling")

    if freqs[0] == 0.0:
        freqs = freqs.copy()
        freqs[0] = float(positive_freqs.min())

    min_freq = float(max(min_freq, positive_freqs.min()))

    log_freqs = np.log10(freqs)
    freq_axis = np.logspace(
        np.log10(min_freq), np.log10(max_freq), half_h, dtype=np.float32
    )
    log_freq_axis = np.log10(freq_axis)
    octaves_from_min = np.log2(freq_axis / min_freq)
    freq_tilt_gain = (10.0 ** ((octaves_from_min * cfg.spectrogram.tilt_db_per_octave) / 20.0)).astype(
        np.float32
    )

    return (
        jnp.asarray(log_freqs, dtype=jnp.float32),
        jnp.asarray(log_freq_axis, dtype=jnp.float32),
        jnp.asarray(freq_tilt_gain, dtype=jnp.float32),
        jnp.asarray(valid_indices, dtype=jnp.int32),
    )


def _make_slice_audio(audio, window, window_size: int, spf: int, n_samples: int):
    """Create a function that returns a windowed stereo segment for a frame index."""

    def slice_audio(frame_idx: jnp.ndarray) -> jnp.ndarray:
        start = frame_idx * spf
        ids = start + jnp.arange(window_size, dtype=jnp.int32)
        ids_clipped = jnp.clip(ids, 0, jnp.int32(n_samples - 1))
        segment = jnp.where(
            (ids[:, None] < n_samples),
            audio[ids_clipped],
            jnp.zeros((window_size, 2), dtype=audio.dtype),
        )
        return segment * window[:, None]

    return slice_audio


def _make_compute_columns(
    slice_audio,
    log_freq_axis,
    log_freqs,
    freq_tilt_gain,
    valid_indices,
    fft_size: int,
    write_px: int,
    gain: float,
):
    """Create a function that maps a frame index to stereo spectrogram columns."""

    def compute_columns(frame_idx: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        windowed = slice_audio(frame_idx)
        # Metal backend requires FFT along the last axis, so operate on a transposed view.
        spectrum = jnp.fft.rfft(windowed.T, n=fft_size, axis=-1).T
        spectrum = jnp.take(spectrum, valid_indices, axis=0)
        spectrum_mag = jnp.maximum(jnp.abs(spectrum), 1e-12)

        peak = jnp.max(spectrum_mag)
        norm = spectrum_mag / jnp.maximum(peak, 1e-6)
        norm = jnp.clip(norm, 0.0, 1.0)

        col_l = jnp.interp(log_freq_axis, log_freqs, norm[:, 0], left=0.0, right=0.0)
        col_r = jnp.interp(log_freq_axis, log_freqs, norm[:, 1], left=0.0, right=0.0)

        col_l = jnp.flip(col_l) * freq_tilt_gain
        col_r = jnp.flip(col_r) * freq_tilt_gain

        col_l = jnp.tile(col_l[:, None] * gain, (1, write_px))
        col_r = jnp.tile(col_r[:, None] * gain, (1, write_px))
        return col_l, col_r

    return compute_columns


def _log_renderer_details(cfg: AppConfig, n_frames: int):
    """Emit a concise description of the renderer configuration when verbose is set."""

    if not cfg.verbose:
        return

    print("🎛 Spectrogram renderer")
    print(f"  frames         : {n_frames}")
    print(f"  window_size    : {cfg.spectrogram.window_size}")
    print(f"  scroll_px      : {cfg.spectrogram.scroll_px}")
    print(f"  write_px       : {cfg.spectrogram.write_px}")
    print(f"  fft_size       : {cfg.spectrogram.fft_size}")
    print(f"  min_freq_hz    : {cfg.spectrogram.min_hz_bound}")
    print(f"  max_freq_hz    : {cfg.spectrogram.max_freq_hz}")
    if cfg.spectrogram.tilt_db_per_octave != 0.0:
        print(f"  tilt_db/octave : {cfg.spectrogram.tilt_db_per_octave}")


def build_renderer_spectrogram_alpha(audio_np: np.ndarray, cfg: AppConfig):
    window_size = int(cfg.spectrogram.window_size)
    fft_size = int(cfg.spectrogram.fft_size)
    spf = max(int(cfg.audio.target_sr // cfg.video.fps), 1)
    scroll_px = max(int(cfg.spectrogram.scroll_px), 1)
    write_px = min(max(int(cfg.spectrogram.write_px), 1), scroll_px)

    audio = jnp.asarray(audio_np, dtype=jnp.float32)
    n_samples = audio.shape[0]

    half_h = cfg.render.render_h // 2
    log_freqs, log_freq_axis, freq_tilt_gain, valid_indices = _compute_frequency_axis(
        cfg, fft_size, half_h
    )

    window = jnp.asarray(_prepare_window(window_size))
    heat_shape = (cfg.render.render_h, cfg.render.render_w)
    heat_zero = jnp.zeros(heat_shape, dtype=jnp.float32)

    gain = float(cfg.scroll.gain)
    decay = float(cfg.scroll.decay)
    reveal_gain = float(cfg.scroll.reveal_gain)
    gamma = float(cfg.scroll.gamma)

    slice_audio = _make_slice_audio(audio, window, window_size, spf, n_samples)
    compute_columns = _make_compute_columns(
        slice_audio,
        log_freq_axis,
        log_freqs,
        freq_tilt_gain,
        valid_indices,
        fft_size,
        write_px,
        gain,
    )

    @jax.jit
    def render_one(frame_idx: jnp.ndarray, heat: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        heat = heat * decay
        heat = jnp.concatenate(
            [heat[:, scroll_px:], jnp.zeros((heat_shape[0], scroll_px), dtype=heat.dtype)],
            axis=1,
        )

        col_l, col_r = compute_columns(frame_idx)

        top = heat[:half_h, :]
        bottom = heat[half_h:, :]

        top = top.at[:, -write_px:].set(jnp.maximum(top[:, -write_px:], col_l))
        bottom = bottom.at[:, -write_px:].set(jnp.maximum(bottom[:, -write_px:], col_r))

        heat = jnp.concatenate([top, bottom], axis=0)
        alpha = 1.0 - jnp.exp(-heat * reveal_gain)
        alpha = jnp.clip(alpha ** gamma, 0.0, 1.0)
        alpha_u8 = (alpha * 255.0).astype(jnp.uint8)
        return heat, alpha_u8

    @jax.jit
    def make_batch(t0: jnp.ndarray, heat: jnp.ndarray):
        def step(carry, i):
            h, _ = carry
            h, a = render_one(t0 + i, h)
            return (h, a), a

        (heat, _), alphas = jax.lax.scan(
            step, (heat, jnp.zeros(heat_shape, dtype=jnp.uint8)), jnp.arange(cfg.render.batch)
        )
        return alphas, heat

    return heat_zero, make_batch, int(np.ceil(audio_np.shape[0] / spf))


class SpectrogramRenderer:
    def __init__(self, audio_np: np.ndarray, cfg: AppConfig):
        self.cfg = cfg
        self.init_heat, self.make_batch, self.n_frames = build_renderer_spectrogram_alpha(
            audio_np.astype(np.float32), cfg
        )
        self.heat = self.init_heat

        _log_renderer_details(cfg, self.n_frames)

        # warmup compile
        alphas, self.heat = self.make_batch(jnp.int32(0), self.heat)
        _ = jax.device_get(alphas)

    def next_alphas(self, t0: int, n: int) -> np.ndarray:
        alphas, self.heat = self.make_batch(jnp.int32(t0), self.heat)
        return np.asarray(jax.device_get(alphas))[:n]
