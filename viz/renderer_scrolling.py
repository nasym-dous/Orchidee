import numpy as np
import jax
import jax.numpy as jnp
from .config import AppConfig


def make_disk_kernel(radius: int):
    r = int(radius)
    ys, xs = np.mgrid[-r:r+1, -r:r+1]
    k = ((xs*xs + ys*ys) <= (r*r)).astype(np.float32)
    s = float(k.sum())
    if s > 0:
        k /= s
    return k


def build_renderer_scrolling_alpha(audio_lr, cfg: AppConfig):
    spf = max(int(cfg.audio.target_sr // cfg.video.fps), 1)
    n_samples = audio_lr.shape[0]

    halfW = cfg.render.render_w // 2
    fps = cfg.video.fps

    # pixels/frame calculé depuis un objectif en secondes
    scroll_px = halfW / (cfg.scroll.seconds_to_center * fps)
    n_new = max(1, int(round(scroll_px)))

    # safety
    n_new = min(n_new, halfW)

    if cfg.verbose:
        print(
            f"🧭 seconds_to_center={cfg.scroll.seconds_to_center:.2f}s -> scroll_px≈{scroll_px:.2f} => n_new={n_new}px/frame")

    if n_new < 1:
        raise ValueError("scroll_px doit être >= 1")

    RENDER_W = cfg.render.render_w
    RENDER_H = cfg.render.render_h
    halfW = RENDER_W // 2

    if n_new > halfW:
        raise ValueError("scroll_px trop grand: doit être <= render_w/2")

    # sample indices dans la fenêtre audio de la frame
    samp_idx = jnp.linspace(0, spf - 1, n_new).astype(jnp.int32)

    # positions X locales (dans chaque demi-écran)
    # gauche: nouvelles colonnes à x=0..n_new-1
    x_new_L = jnp.arange(0, n_new, dtype=jnp.int32)
    # droite: nouvelles colonnes à la fin du demi-écran
    x_new_R = jnp.arange(halfW - n_new, halfW, dtype=jnp.int32)

    # conv kernel HWIO
    disk = make_disk_kernel(cfg.scroll.line_thickness)
    if cfg.verbose:
        print(f"🧱 Conv kernel: {disk.shape} (radius={cfg.scroll.line_thickness}, sum={disk.sum():.3f})")
    K = jnp.asarray(disk)[:, :, None, None]

    DECAY = float(cfg.scroll.decay)
    GAIN = float(cfg.scroll.gain)
    REVEAL_GAIN = float(cfg.scroll.reveal_gain)
    GAMMA = float(cfg.scroll.gamma)

    def init_heat():
        # heat full frame, mais on la manipule en 2 moitiés
        return jnp.zeros((RENDER_H, RENDER_W), jnp.float32)

    @jax.jit
    def render_one(frame_idx, heat):
        # 1) decay
        heat = heat * DECAY

        # 2) split en deux moitiés
        heatL = heat[:, :halfW]
        heatR = heat[:, halfW:]

        # 3) scrolling vers le centre
        # Gauche: shift RIGHT => on pousse l'historique vers le centre
        heatL = jnp.concatenate(
            [jnp.zeros((RENDER_H, n_new), dtype=heatL.dtype), heatL[:, :-n_new]],
            axis=1
        )
        # Droite: shift LEFT => on pousse l'historique vers le centre
        heatR = jnp.concatenate(
            [heatR[:, n_new:], jnp.zeros((RENDER_H, n_new), dtype=heatR.dtype)],
            axis=1
        )

        # 4) sample audio
        start = jnp.clip(frame_idx * spf, 0, n_samples - 1)
        ids = jnp.clip(start + samp_idx, 0, n_samples - 1)
        lr = audio_lr[ids]  # (n_new, 2)

        L = lr[:, 0] * GAIN
        R = lr[:, 1] * GAIN

        # 5) mapping Y: pleine hauteur
        cy = (RENDER_H - 1) * 0.5
        amp = (RENDER_H - 1) * 0.5  # pleine hauteur (0..H-1)
        yL = jnp.clip((cy - L * amp).astype(jnp.int32), 0, RENDER_H - 1)
        yR = jnp.clip((cy - R * amp).astype(jnp.int32), 0, RENDER_H - 1)

        # 6) hits locaux (dans chaque moitié)
        hitsL = jnp.zeros((RENDER_H, halfW), dtype=jnp.float32)
        hitsR = jnp.zeros((RENDER_H, halfW), dtype=jnp.float32)

        # Gauche: on écrit sur le bord gauche
        hitsL = hitsL.at[yL, x_new_L].add(1.0)
        # Droite: on écrit sur le bord droit du demi-écran
        hitsR = hitsR.at[yR, x_new_R].add(1.0)

        # 7) épaissir via convolution (séparément)
        hitsL4 = hitsL[None, :, :, None]
        hitsR4 = hitsR[None, :, :, None]

        thickL4 = jax.lax.conv_general_dilated(
            hitsL4, K,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        thickR4 = jax.lax.conv_general_dilated(
            hitsR4, K,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )

        heatL = heatL + thickL4[0, :, :, 0]
        heatR = heatR + thickR4[0, :, :, 0]

        heat = jnp.concatenate([heatL, heatR], axis=1)

        # 8) alpha (sans max global)
        alpha = 1.0 - jnp.exp(-heat * REVEAL_GAIN)
        alpha = jnp.clip(alpha ** GAMMA, 0.0, 1.0)
        alpha_u8 = (alpha * 255.0).astype(jnp.uint8)
        return heat, alpha_u8

    @jax.jit
    def make_batch(t0, heat):
        def step(h, i):
            h, a = render_one(t0 + i, h)
            return h, a
        heat, alphas = jax.lax.scan(step, heat, jnp.arange(cfg.render.batch, dtype=jnp.int32))
        return alphas, heat

    return init_heat, make_batch



class ScrollingRenderer:
    def __init__(self, audio_np: np.ndarray, cfg: AppConfig):
        self.cfg = cfg
        self.audio = jnp.asarray(audio_np)

        self.spf = max(int(cfg.audio.target_sr // cfg.video.fps), 1)
        self.n_frames = int(np.ceil(audio_np.shape[0] / self.spf))

        if cfg.verbose:
            print("🎛 Renderer config")
            print(f"  frames         : {self.n_frames}")
            print(f"  spf            : {self.spf}")
            print(f"  alpha internal : {cfg.render.render_w}x{cfg.render.render_h}")
            print(f"  scroll_px      : {cfg.scroll.scroll_px}")

        self.init_heat, self.make_batch = build_renderer_scrolling_alpha(self.audio, cfg)
        self.heat = self.init_heat()

        # warmup compile
        alphas, self.heat = self.make_batch(jnp.int32(0), self.heat)
        _ = jax.device_get(alphas)

    def next_alphas(self, t0: int, n: int) -> np.ndarray:
        alphas, self.heat = self.make_batch(jnp.int32(t0), self.heat)
        return np.asarray(jax.device_get(alphas))[:n]
