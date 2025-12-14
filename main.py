from viz.config import AppConfig
from viz.diagnostics import print_env_diagnostics
from viz.io_audio import load_audio_stereo_ffmpeg
from viz.io_cover import load_cover_bgr
from viz.renderer_scrolling import ScrollingRenderer
from viz.renderer_spectrogram import SpectrogramRenderer
from viz.compositor import compose_frame_from_alpha
from viz.pipeline import run_pipeline
from viz.encode import mux_audio
from viz.stats import Timer


def main():
    cfg = AppConfig.default()
    cfg.apply_env()
    cfg.validate()

    print_env_diagnostics(cfg)

    with Timer("audio decode"):
        audio, sr = load_audio_stereo_ffmpeg(cfg.audio.audio_path, cfg.audio.target_sr, verbose=cfg.verbose)
        if cfg.verbose:
            print(f"🎵 sr={sr} | samples={audio.shape[0]} | duration={audio.shape[0]/sr:.2f}s")

    with Timer("cover load"):
        cover = load_cover_bgr(cfg.video.w, cfg.video.h, cfg.paths.cover_path)

    with Timer("init renderer (JIT warmup)"):
        if cfg.visual_mode == "spectrogram":
            renderer = SpectrogramRenderer(audio, cfg)
        else:
            renderer = ScrollingRenderer(audio, cfg)

    with Timer("video render (AVI)"):
        out_avi = run_pipeline(
            cfg=cfg,
            renderer=renderer,
            cover_bgr=cover,
            compositor=compose_frame_from_alpha,
        )

    with Timer("mux audio"):
        mux_audio(
            in_avi=out_avi,
            audio_path=cfg.audio.audio_path,
            out_mp4=cfg.paths.out_final,
            verbose=cfg.verbose,
        )

    print(f"✅ FINAL OK → {cfg.paths.out_final}")


if __name__ == "__main__":
    main()
