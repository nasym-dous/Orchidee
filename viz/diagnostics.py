import jax
from .config import AppConfig


def print_env_diagnostics(cfg: AppConfig):
    print("backend:", jax.default_backend())
    print("devices:", jax.devices())
    if cfg.verbose:
        print("🧩 Config summary")
        print(f"  OUT  : {cfg.video.w}x{cfg.video.h} @ {cfg.video.fps} fps")
        print(f"  ALPHA: {cfg.render.render_w}x{cfg.render.render_h} | batch={cfg.render.batch} | q={cfg.render.max_buffer_batches}")
        if cfg.visual_mode == "scroll":
            print(f"  MODE scroll: px/frame={cfg.scroll.scroll_px} | thickness={cfg.scroll.line_thickness} | decay={cfg.scroll.decay}")
        else:
            print(
                "  MODE spectrogram: "
                f"n_fft={cfg.spectrogram.n_fft} | "
                f"dyn={cfg.spectrogram.dynamic_range}dB | "
                f"gamma={cfg.spectrogram.gamma} | gain={cfg.spectrogram.gain}"
            )
