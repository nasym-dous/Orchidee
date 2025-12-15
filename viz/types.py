from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class AudioChunk:
    """Stereo audio data consumed by the renderer stage."""

    samples: np.ndarray
    sample_rate: int


@dataclass
class AlphaBatch:
    """Batch of low-res alpha masks produced by the renderer stage."""

    start_frame: int
    alphas: np.ndarray
    total_frames: int | None = None


@dataclass
class FrameBatch:
    """Batch of BGR frames ready for encoding."""

    start_frame: int
    frames: List[np.ndarray]
    total_frames: int | None = None
