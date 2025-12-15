import numpy as np
import jax
import jax.numpy as jnp
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
        self.log_freq_axis = np.log10(self.freq_axis)
        octaves_from_min = np.log2(self.freq_axis / self.min_freq)
        self.freq_tilt_gain = (10.0 ** ((octaves_from_min * self.tilt_db_per_octave) / 20.0)).astype(
            np.float32
        )
        self.heat = np.zeros((self.h, self.w), dtype=np.float32)

        self.columns_to_generate = int(np.ceil(self.scroll_px / max(self.write_px, 1)))

        # GPU friendly buffers/constants
        padded_audio = np.pad(self.audio, ((0, self.window_size), (0, 0)))
        self.audio_jnp = jnp.asarray(padded_audio)
        self.window_jnp = jnp.asarray(self.window)
        self.freqs_jnp = jnp.asarray(self.freqs)
        self.log_freqs_jnp = jnp.asarray(self.log_freqs)
        self.log_freq_axis_jnp = jnp.asarray(self.log_freq_axis)
        self.freq_tilt_gain_jnp = jnp.asarray(self.freq_tilt_gain)
        self.valid_bins_jnp = jnp.asarray(self.valid_bins)

        self.columns_idx = jnp.arange(self.columns_to_generate, dtype=jnp.int32)
        self._build_gpu_renderer()

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

    def _build_gpu_renderer(self) -> None:
        cfg = self.cfg
        spf = self.spf
        scroll_px = int(self.scroll_px)
        write_px = int(self.write_px)
        columns_to_generate = int(self.columns_to_generate)
        window_size = int(self.window_size)
        fft_size = int(self.fft_size)
        h = int(self.h)
        half_h = int(self.half_h)
        gain = float(cfg.scroll.gain)
        decay = float(cfg.scroll.decay)
        reveal_gain = float(cfg.scroll.reveal_gain)
        gamma = float(cfg.scroll.gamma)

        audio = self.audio_jnp
        window = self.window_jnp
        freq_tilt_gain = self.freq_tilt_gain_jnp
        log_freq_axis = self.log_freq_axis_jnp
        log_freqs = self.log_freqs_jnp
        valid_bins = self.valid_bins_jnp
        col_indices = self.columns_idx

        audio_len = audio.shape[0]
        max_start = jnp.int32(audio_len - window_size)

        def slice_window(start_idx: jnp.ndarray) -> jnp.ndarray:
            safe_start = jnp.minimum(jnp.maximum(start_idx, 0), max_start)
            return jax.lax.dynamic_slice(audio, (safe_start, 0), (window_size, 2))

        def make_columns(t0: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            start_sample = t0 * spf
            hop = spf / jnp.maximum(columns_to_generate, 1)
            starts = start_sample + hop * col_indices.astype(jnp.float32)
            start_ids = jnp.floor(starts).astype(jnp.int32)

            windows = jax.vmap(slice_window)(start_ids)
            windows = windows * window[None, :, None]

            spectrum = jnp.fft.rfft(windows, n=fft_size, axis=1)[:, valid_bins]
            mag = jnp.maximum(jnp.abs(spectrum).astype(jnp.float32), 1e-12)

            peaks = jnp.max(mag, axis=(1, 2))
            peaks = jnp.where(peaks > 0.0, peaks, 1.0)
            norm = jnp.clip(mag / peaks[:, None, None], 0.0, 1.0)

            def interp_channel(arr):
                return jnp.interp(log_freq_axis, log_freqs, arr, left=0.0, right=0.0)

            col_l = jax.vmap(lambda a: interp_channel(a[:, 0]))(norm)
            col_r = jax.vmap(lambda a: interp_channel(a[:, 1]))(norm)

            if self.tilt_db_per_octave != 0.0:
                col_l = col_l * freq_tilt_gain
                col_r = col_r * freq_tilt_gain

            col_l = col_l[:, ::-1]
            col_r = col_r[:, ::-1]

            def write_columns(i, imgs):
                img_l, img_r = imgs
                start_px = i * write_px
                end_px = jnp.minimum(start_px + write_px, scroll_px)
                def do_write(img_l, img_r):
                    slice_l = col_l[i][:, None]
                    slice_r = col_r[i][:, None]
                    img_l = jax.lax.dynamic_update_slice(img_l, slice_l, (0, start_px))
                    img_r = jax.lax.dynamic_update_slice(img_r, slice_r, (0, start_px))
                    return img_l, img_r
                img_l, img_r = jax.lax.cond(
                    start_px < scroll_px, do_write, lambda a, b: (a, b), img_l, img_r
                )
                return img_l, img_r

            init_imgs = (
                jnp.zeros((half_h, scroll_px), dtype=jnp.float32),
                jnp.zeros((half_h, scroll_px), dtype=jnp.float32),
            )
            col_img_l, col_img_r = jax.lax.fori_loop(0, columns_to_generate, write_columns, init_imgs)
            return col_img_l * gain, col_img_r * gain

        def render_one(t0: jnp.ndarray, heat: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            heat = heat * decay
            heat = jnp.concatenate([heat[:, self.scroll_px :], jnp.zeros((h, self.scroll_px), heat.dtype)], axis=1)

            col_l, col_r = make_columns(t0)

            top = heat[:half_h]
            bottom = heat[half_h:]

            top = top.at[:, -scroll_px:].set(jnp.maximum(top[:, -scroll_px:], col_l))
            bottom = bottom.at[:, -scroll_px:].set(jnp.maximum(bottom[:, -scroll_px:], col_r))

            heat = jnp.concatenate([top, bottom], axis=0)

            alpha = 1.0 - jnp.exp(-heat * reveal_gain)
            alpha = jnp.clip(alpha ** gamma, 0.0, 1.0)
            alpha_u8 = (alpha * 255.0).astype(jnp.uint8)
            return heat, alpha_u8

        @jax.jit
        def render_batch(t0: jnp.ndarray, heat: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            def step(carry, idx):
                h = carry
                h, alpha = render_one(t0 + idx, h)
                return h, alpha

            heat, alphas = jax.lax.scan(step, heat, jnp.arange(cfg.render.batch, dtype=jnp.int32))
            return heat, alphas

        self._render_one = render_one
        self._render_batch = render_batch
        self.heat = jnp.zeros((h, self.w), dtype=jnp.float32)
        _, warm = self._render_batch(jnp.int32(0), self.heat)
        _ = jax.device_get(warm)

    def next_alphas(self, t0: int, n: int) -> np.ndarray:
        heat, alphas = self._render_batch(jnp.int32(t0), self.heat)
        self.heat = heat
        return np.asarray(jax.device_get(alphas))[:n]
