# ======================
# GLOBAL VERBOSE SWITCH
# ======================
VERBOSE = True   # mets False quand tout est stable


# ======================
# ENV / LIB VERBOSE
# ======================
import os
os.environ["ENABLE_PJRT_COMPATIBILITY"] = "1"

if VERBOSE:
    os.environ["JAX_LOG_COMPILES"] = "1"
    os.environ["JAX_DEBUG_NANS"] = "1"
    os.environ["JAX_TRACEBACK_FILTERING"] = "off"
else:
    os.environ["JAX_TRACEBACK_FILTERING"] = "on"


# ======================
# Imports
# ======================
import time
import threading
import queue
import subprocess
import cv2
import numpy as np
import jax
import jax.numpy as jnp

# RAM
try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False
import resource


# ======================
# Config
# ======================
AUDIO_PATH = "music.mp3"
COVER_PATH = "cover.jpeg"
OUT_AVI = "out.avi"
OUT_FINAL = "out_with_audio.mp4"

# Output video resolution
W, H = 1080, 1080
FPS = 60

# Internal render resolution (alpha mask)
RENDER_W, RENDER_H = 480, 480   # 1024 -> rapide ; 1536/2048 -> plus fin

# Pipeline buffering
BATCH = 8
MAX_BUFFER_BATCHES = 8
TARGET_SR = 48000

# ---- Visibility / YouTube safe params ----
LINE_THICKNESS = 2          # IMPORTANT: int (kernel conv)
LINE_SAMPLES = 4            # points interpolés par segment
BASELINE = 0.12             # 0.08–0.18
GLOW_SIGMA = 1.2            # glow low-res (0 = off)

# Imager params
PTS = 1024                  # 1024/2048 (4096 coûte cher)
DECAY = 0.0
GAIN = 0.5                  # pas d’auto-gain → ajuste ici
REVEAL_GAIN = 1           # k du 1-exp(-k*heat)
GAMMA = 1


# ======================
# OpenCV perf hint
# ======================
# (0 = laisser OpenCV gérer; sur mac ça permet souvent multi-core)
try:
    cv2.setNumThreads(0)
except Exception:
    pass


# ======================
# Timer
# ======================
class Timer:
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.t0 = time.perf_counter()
    def __exit__(self, *_):
        print(f"⏱️ {self.name}: {time.perf_counter() - self.t0:.3f}s")


# ======================
# RAM helpers
# ======================
if _HAS_PSUTIL:
    _proc = psutil.Process(os.getpid())
    def ram_mb():
        return _proc.memory_info().rss / (1024 ** 2)
else:
    # ru_maxrss: KB on macOS
    def ram_mb():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# ======================
# Diagnostics helpers
# ======================
def diagnose_audio(audio, sr):
    print("🎵 Audio diagnostics")
    print(f"  sr        : {sr}")
    print(f"  samples   : {audio.shape[0]}")
    print(f"  duration  : {audio.shape[0] / sr:.2f}s")
    print(f"  channels  : {audio.shape[1]}")
    if not np.isfinite(audio).all():
        raise ValueError("Audio contient NaN/Inf")

def diagnose_cover(cover):
    print("🖼️ Cover diagnostics")
    print(f"  shape : {cover.shape}")
    print(f"  dtype : {cover.dtype}")
    if cover.shape != (H, W, 3):
        raise ValueError("Cover mal redimensionnée")

def estimate_queue_ram_gb():
    # queue stores alpha batches (BATCH, RENDER_H, RENDER_W) uint8
    bytes_per_alpha = RENDER_W * RENDER_H
    total = bytes_per_alpha * BATCH * MAX_BUFFER_BATCHES
    return total / (1024**3)


# ======================
# Audio loader (ffmpeg)
# ======================
def load_audio_stereo_ffmpeg(path, target_sr):
    cmd = [
        "ffmpeg", "-v", "info" if VERBOSE else "error",
        "-i", path,
        "-vn",
        "-ac", "2",
        "-ar", str(target_sr),
        "-f", "f32le",
        "pipe:1"
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, check=True)
    audio = np.frombuffer(p.stdout, dtype=np.float32).reshape((-1, 2))
    return audio, target_sr


# ======================
# Cover loader
# ======================
def load_cover_bgr(path):
    cover = cv2.imread(path, cv2.IMREAD_COLOR)
    if cover is None:
        raise FileNotFoundError(f"Cover introuvable: {path}")
    cover = cv2.resize(cover, (W, H), interpolation=cv2.INTER_AREA)
    return cover


# ======================
# Thickness kernel (disk) for convolution
# ======================
def make_disk_kernel(radius: int):
    r = int(radius)
    ys, xs = np.mgrid[-r:r+1, -r:r+1]
    k = ((xs*xs + ys*ys) <= (r*r)).astype(np.float32)
    # normalisation légère (évite que ça explose trop vite)
    s = float(k.sum())
    if s > 0:
        k /= s
    return k

DISK_K = make_disk_kernel(LINE_THICKNESS)
if VERBOSE:
    print(f"🧱 Conv kernel: {DISK_K.shape} (radius={LINE_THICKNESS}, sum={DISK_K.sum():.3f})")


# ======================
# JAX renderer: returns ALPHA ONLY (low-res)
# - thickness via convolution (fast)
# - alpha = 1 - exp(-k*heat) (no max-reduction)
# ======================
def build_renderer_alpha(audio_lr):
    spf = max(int(TARGET_SR // FPS), 1)
    n_samples = audio_lr.shape[0]

    # sample indices inside each frame
    idx = jnp.linspace(0, spf - 1, PTS).astype(jnp.int32)
    eps = 1e-6

    # convolution kernel: HWIO
    K = jnp.asarray(DISK_K)
    K = K[:, :, None, None]  # (kH,kW,1,1)

    def init_heat():
        return jnp.zeros((RENDER_H, RENDER_W), jnp.float32)

    @jax.jit
    def render_one(frame_idx, heat):
        start = jnp.clip(frame_idx * spf, 0, n_samples - 1)
        ids = jnp.clip(start + idx, 0, n_samples - 1)
        lr = audio_lr[ids]

        L, R = lr[:, 0], lr[:, 1]
        mid = (L + R) * 0.70710678
        side = (R - L) * 0.70710678

        # ✅ NO auto-gain: fixed gain only
        mid = mid * GAIN
        side = side * GAIN

        # map [-1,1] -> low-res pixels
        x = (side * 0.5 + 0.5) * (RENDER_W - 1)
        y = (0.5 - mid * 0.5) * (RENDER_H - 1)
        xi = jnp.clip(x.astype(jnp.int32), 0, RENDER_W - 1)
        yi = jnp.clip(y.astype(jnp.int32), 0, RENDER_H - 1)

        # persistence
        heat = heat * DECAY

        # connect points with segments
        x0, y0 = xi[:-1], yi[:-1]
        x1, y1 = xi[1:], yi[1:]

        t = jnp.linspace(0.0, 1.0, LINE_SAMPLES)
        xs = x0[:, None] * (1.0 - t) + x1[:, None] * t
        ys = y0[:, None] * (1.0 - t) + y1[:, None] * t

        xs = xs.reshape(-1)
        ys = ys.reshape(-1)

        # scatter thin hits
        xs_i = jnp.clip(xs, 0, RENDER_W - 1).astype(jnp.int32)
        ys_i = jnp.clip(ys, 0, RENDER_H - 1).astype(jnp.int32)

        hits = jnp.zeros((RENDER_H, RENDER_W), dtype=jnp.float32)
        hits = hits.at[ys_i, xs_i].add(1.0)

        # thicken via convolution (fast)
        hits4 = hits[None, :, :, None]  # NHWC
        thick4 = jax.lax.conv_general_dilated(
            hits4, K,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        thick = thick4[0, :, :, 0]

        heat = heat + thick

        # alpha without global max()
        alpha = 1.0 - jnp.exp(-heat * REVEAL_GAIN)
        alpha = jnp.clip(alpha ** GAMMA, 0.0, 1.0)
        alpha_u8 = (alpha * 255.0).astype(jnp.uint8)

        return heat, alpha_u8

    @jax.jit
    def make_batch(t0, heat):
        def step(h, i):
            h, a = render_one(t0 + i, h)
            return h, a
        heat, alphas = jax.lax.scan(step, heat, jnp.arange(BATCH, dtype=jnp.int32))
        return alphas, heat  # (BATCH, RENDER_H, RENDER_W)

    return init_heat, make_batch


# ======================
# Renderer class
# ======================
class ImagerRenderer:
    def __init__(self, audio_np):
        self.BATCH = BATCH
        self.audio = jnp.asarray(audio_np)

        self.spf = max(int(TARGET_SR // FPS), 1)
        self.N_FRAMES = int(np.ceil(audio_np.shape[0] / self.spf))

        if VERBOSE:
            print("🎛 Renderer config")
            print(f"  frames         : {self.N_FRAMES}")
            print(f"  spf            : {self.spf}")
            print(f"  alpha internal : {RENDER_W}x{RENDER_H}")
            print(f"  queue worst    : ~{estimate_queue_ram_gb():.3f} GB (alphas)")

        self.init_heat, self.make_batch = build_renderer_alpha(self.audio)
        self.heat = self.init_heat()

        # warmup compile
        alphas, self.heat = self.make_batch(jnp.int32(0), self.heat)
        _ = jax.device_get(alphas)

    def next_alphas(self, t0, n):
        alphas, self.heat = self.make_batch(jnp.int32(t0), self.heat)
        a = np.asarray(jax.device_get(alphas))
        return a[:n]  # (n, RENDER_H, RENDER_W) uint8


# ======================
# Compose 4K frame on CPU (fast-ish)
# - glow is done in LOW-RES before upscale (cheap)
# ======================
def compose_frame_from_alpha(cover_bgr_u8: np.ndarray, alpha_u8_small: np.ndarray) -> np.ndarray:
    # glow in low-res (avoid 4K blur!)
    if GLOW_SIGMA and GLOW_SIGMA > 0:
        alpha_u8_small = cv2.GaussianBlur(
            alpha_u8_small, (0, 0),
            sigmaX=float(GLOW_SIGMA), sigmaY=float(GLOW_SIGMA)
        )

    # upscale alpha once
    alpha_u8 = cv2.resize(alpha_u8_small, (W, H), interpolation=cv2.INTER_LINEAR)

    # fast baseline blend in integer domain
    baseline_u8 = int(BASELINE * 255)
    a16 = alpha_u8.astype(np.uint16)
    alpha2_u8 = baseline_u8 + ((a16 * (255 - baseline_u8)) >> 8)
    alpha2_u8 = alpha2_u8.astype(np.uint8)

    # cover * alpha2 / 255 (uint8 math)
    frame = (cover_bgr_u8.astype(np.uint16) * alpha2_u8[..., None].astype(np.uint16) // 255).astype(np.uint8)
    return frame


# ======================
# Mux audio
# ======================
def mux_audio():
    cmd = [
        "ffmpeg", "-y",
        "-i", OUT_AVI,
        "-i", AUDIO_PATH,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "18",
        "-c:a", "aac",
        "-shortest",
        OUT_FINAL
    ]
    subprocess.run(cmd, check=True)


# ======================
# Main
# ======================
def main():
    print("backend:", jax.default_backend())
    print("devices:", jax.devices())

    with Timer("audio decode"):
        audio, sr = load_audio_stereo_ffmpeg(AUDIO_PATH, TARGET_SR)
        if VERBOSE:
            diagnose_audio(audio, sr)

    with Timer("cover load"):
        cover = load_cover_bgr(COVER_PATH)
        if VERBOSE:
            diagnose_cover(cover)

    with Timer("init renderer (JIT warmup)"):
        renderer = ImagerRenderer(audio)

    q = queue.Queue(MAX_BUFFER_BATCHES)
    STOP = object()

    fps_counter = {"frames": 0, "t0": None, "t1": None}

    def producer():
        if VERBOSE:
            print("🚀 Producer started")
        t = 0
        while t < renderer.N_FRAMES:
            n = min(renderer.BATCH, renderer.N_FRAMES - t)

            t_batch0 = time.perf_counter()
            alphas = renderer.next_alphas(t, n)
            dt = time.perf_counter() - t_batch0
            if VERBOSE and (t == 0 or t % (FPS * 5) == 0):
                print(f"🧠 Producer batch {n} frames: {dt:.3f}s => {n / max(dt,1e-6):.1f} fps (producer)")

            q.put(alphas)
            t += n

        q.put(STOP)

    def consumer():
        out = cv2.VideoWriter(OUT_AVI, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
        if not out.isOpened():
            raise RuntimeError("VideoWriter non ouvert")

        fps_counter["t0"] = time.perf_counter()
        written = 0

        while True:
            item = q.get()
            if item is STOP:
                break

            for alpha_small in item:
                frame = compose_frame_from_alpha(cover, alpha_small)
                out.write(frame)
                written += 1
                fps_counter["frames"] += 1

                if VERBOSE and written % (FPS * 5) == 0:
                    elapsed = time.perf_counter() - fps_counter["t0"]
                    avg_fps = written / max(elapsed, 1e-6)
                    print(f"📼 {written} frames | avg FPS ≈ {avg_fps:.1f} | RAM ≈ {ram_mb():.0f} MB")

            q.task_done()

        out.release()
        fps_counter["t1"] = time.perf_counter()

    with Timer("video render (AVI)"):
        tp = threading.Thread(target=producer, name="producer")
        tc = threading.Thread(target=consumer, name="consumer")
        tc.start(); tp.start()
        tp.join(); tc.join()

    total_time = fps_counter["t1"] - fps_counter["t0"]
    total_frames = fps_counter["frames"]
    avg_fps = total_frames / max(total_time, 1e-6)
    print("🚀 Render performance")
    print(f"  frames rendered : {total_frames}")
    print(f"  total time     : {total_time:.2f}s")
    print(f"  avg FPS        : {avg_fps:.2f}")
    print(f"  RAM (now)      : {ram_mb():.0f} MB")
    print(f"✅ Vidéo MJPG terminée : {OUT_AVI}")

    with Timer("mux audio"):
        mux_audio()

    print(f"✅ FINAL OK → {OUT_FINAL}")


if __name__ == "__main__":
    main()
