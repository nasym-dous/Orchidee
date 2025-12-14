import cv2
import numpy as np
from .config import AppConfig


def compose_frame_from_alpha(cover_bgr_u8: np.ndarray, alpha_u8_small: np.ndarray, cfg: AppConfig) -> np.ndarray:
    if cfg.render.glow_sigma and cfg.render.glow_sigma > 0:
        alpha_u8_small = cv2.GaussianBlur(
            alpha_u8_small, (0, 0),
            sigmaX=float(cfg.render.glow_sigma),
            sigmaY=float(cfg.render.glow_sigma),
        )

    alpha_u8 = cv2.resize(alpha_u8_small, (cfg.video.w, cfg.video.h), interpolation=cv2.INTER_LINEAR)

    # background baseline (image toujours un peu visible)
    baseline_u8 = int(cfg.render.baseline * 255)
    bg = (cover_bgr_u8.astype(np.uint16) * baseline_u8 // 255).astype(np.uint16)

    # trace blanche pure (additive)
    white = alpha_u8.astype(np.uint16)  # 0..255
    frame = bg + white[..., None]
    frame = np.clip(frame, 0, 255).astype(np.uint8)

    if cfg.render.draw_center_lines:
        cy = cfg.video.h // 2
        cv2.line(frame, (0, cy), (cfg.video.w - 1, cy), (18, 18, 18), 1)
        cx = cfg.video.w // 2
        cv2.line(frame, (cx, 0), (cx, cfg.video.h - 1), (18, 18, 18), 1)

    return frame
