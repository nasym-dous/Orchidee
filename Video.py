import cv2


def build_fourcc(codec: str = "MJPG"):
    """Return a FOURCC code compatible with the installed OpenCV version.

    Some OpenCV wheels expose ``VideoWriter_fourcc`` at the top level while
    older releases use ``cv2.cv.CV_FOURCC``. This helper picks the available
    implementation so the rest of the code does not fail to import on
    environments with a different OpenCV layout.
    """

    if hasattr(cv2, "VideoWriter_fourcc"):
        return cv2.VideoWriter_fourcc(*codec)

    legacy_fourcc = getattr(getattr(cv2, "cv", None), "CV_FOURCC", None)
    if legacy_fourcc is not None:
        return legacy_fourcc(*codec)

    raise AttributeError("No FOURCC constructor found in the installed OpenCV package")


class Video:
    def __init__(self, fps, length, factor, fileName):
        self.fps = fps
        self.length = length
        self.factor = factor
        self.height = 240*factor
        self.width = 480*factor
        self.fileName = fileName

        fourcc = build_fourcc("MJPG")
        self.video = cv2.VideoWriter(f'./{fileName}.avi', fourcc, fps, (self.width, self.height))
