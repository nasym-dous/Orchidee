from dataclasses import dataclass, field
import os


@dataclass
class PathConfig:
    audio_path: str = "0481 v41B.mp3"
    cover_path: str = "cover.jpeg"
    out_avi: str = "out.avi"
    out_final: str = "out_with_audio.mp4"


@dataclass
class AudioConfig:
    audio_path: str = "music.mp3"
    target_sr: int = 48000


@dataclass
class VideoConfig:
    w: int = 1440
    h: int = 1440
    fps: int = 60
    fourcc: str = "MJPG"


@dataclass
class RenderConfig:
    render_w: int = 1440
    render_h: int = 1440
    batch: int = 8
    max_buffer_batches: int = 8

    # low-res alpha look
    baseline: float = 0.12
    glow_sigma: float = 1.0
    draw_center_lines: bool = False


@dataclass
class ScrollConfig:
    seconds_to_center: float = 0.2
    # pixels per frame to scroll left (and number of new columns written at right)
    scroll_px: int = 6

    # thickness is conv kernel radius
    line_thickness: int = 1

    # trace dynamics
    decay: float = 1.0
    gain: float = 1.0
    reveal_gain: float = 2.0
    gamma: float = 1

    # stereo layout
    stereo_split: bool = True  # L in top half, R in bottom half


@dataclass
class SpectrogramConfig:
    max_freq_hz: float = 24_000.0
    scroll_px: int = 4
    window_size: int = 2**14
    fft_size: int = 2**14
    floor_db: float = -15.0
    ceiling_db: float = 0.0
    pre_emphasis: float = 0.99
    denoise_reduction_db: float = 0


@dataclass
class AppConfig:
    verbose: bool = True  # controls application logs
    verbose_lib: bool = False  # controls noisy third-party tools (ffmpeg, etc.)

    paths: PathConfig = field(default_factory=PathConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    scroll: ScrollConfig = field(default_factory=ScrollConfig)
    spectrogram: SpectrogramConfig = field(default_factory=SpectrogramConfig)

    @staticmethod
    def default() -> "AppConfig":
        cfg = AppConfig()
        cfg.audio.audio_path = cfg.paths.audio_path
        return cfg


    def apply_env(self):
        os.environ["ENABLE_PJRT_COMPATIBILITY"] = "1"
        if self.verbose:
            os.environ["JAX_DEBUG_NANS"] = "1"
            os.environ["JAX_TRACEBACK_FILTERING"] = "off"
        else:
            os.environ["JAX_TRACEBACK_FILTERING"] = "on"

    def validate(self):
        assert self.video.w > 0 and self.video.h > 0
        assert self.video.fps > 0
        assert self.render.render_w > 0 and self.render.render_h > 0
        assert self.render.render_w % 2 == 0, "render_w doit être pair (split stéréo propre)"
        assert self.render.batch >= 1
        assert self.render.max_buffer_batches >= 1
        assert self.audio.target_sr >= 8000
        assert self.scroll.seconds_to_center >= 0.1
        assert self.scroll.scroll_px >= 1
        assert self.scroll.line_thickness >= 0
        assert 0.0 <= self.render.baseline <= 0.4
        assert 0.0 <= self.scroll.decay <= 1.0
        assert self.scroll.reveal_gain > 0.0
        assert self.scroll.gamma > 0.0
        assert self.spectrogram.max_freq_hz > 0
        assert self.spectrogram.scroll_px >= 1
        assert self.spectrogram.scroll_px <= self.render.render_w
        assert self.spectrogram.window_size > 0
        assert self.spectrogram.fft_size >= self.spectrogram.window_size
        assert self.spectrogram.floor_db < self.spectrogram.ceiling_db
        assert 0.0 <= self.spectrogram.pre_emphasis < 1.0
        assert self.spectrogram.denoise_reduction_db >= 0.0
