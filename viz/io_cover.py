import cv2


def load_cover_bgr(w: int, h: int, path: str):
    cover = cv2.imread(path, cv2.IMREAD_COLOR)
    if cover is None:
        raise FileNotFoundError(f"Cover introuvable: {path}")
    cover = cv2.resize(cover, (w, h), interpolation=cv2.INTER_AREA)
    return cover
