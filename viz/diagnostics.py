import jax
from .config import AppConfig


def print_env_diagnostics(cfg: AppConfig):
    print("backend:", jax.default_backend())
    print("devices:", jax.devices())
    if cfg.verbose:
        print("🧩 Config summary")
        print(f"  OUT  : {cfg.video.w}x{cfg.video.h} @ {cfg.video.fps} fps")
        print(f"  ALPHA: {cfg.render.render_w}x{cfg.render.render_h} | batch={cfg.render.batch} | q={cfg.render.max_buffer_batches}")
        halfW = cfg.render.render_w // 2
        fps = cfg.video.fps
        target_scroll_px = halfW / (cfg.scroll.seconds_to_center * fps)
        shift_px = min(max(1, int(round(target_scroll_px))), halfW)
        write_px = min(max(1, int(cfg.scroll.write_px)), shift_px)

        print(
            f"  shift_px={shift_px} | write_px={write_px} | thickness={cfg.scroll.line_thickness} | decay={cfg.scroll.decay}")
