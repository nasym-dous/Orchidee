from viz.config import AppConfig
from viz.diagnostics import print_env_diagnostics
from viz.io_cover import load_cover_bgr
from viz.pipeline import run_pipeline
from viz.stats import Timer


def main():
    cfg = AppConfig.default()
    cfg.apply_env()
    cfg.validate()

    print_env_diagnostics(cfg)

    with Timer("cover load"):
        cover = load_cover_bgr(cfg.video.w, cfg.video.h, cfg.paths.cover_path)

    with Timer("video pipeline"):
        out_final = run_pipeline(cfg=cfg, cover_bgr=cover)

    print(f"✅ FINAL OK → {out_final}")


if __name__ == "__main__":
    main()
