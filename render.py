"""Render two vertical (1080x1920) animated reels for Instagram.

reel1_hyperspace.mp4  - psychedelic warp tunnel with beat-synced pulses
reel2_kaleido.mp4     - neon kaleidoscope fractal bloom

Usage: python3 render.py
Requires numpy, Pillow and ffmpeg.
"""
import subprocess, wave, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 540, 960          # internal render size (upscaled 2x by ffmpeg)
FPS, DUR = 30, 12
N = FPS * DUR
BPM = 128
BEAT = 60 / BPM

y, x = np.mgrid[0:H, 0:W].astype(np.float32)
X = (x - W / 2) / (W / 2)
Y = (y - H / 2) / (W / 2)
R = np.sqrt(X * X + Y * Y) + 1e-4
A = np.arctan2(Y, X)


def font(size):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"]:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


def beat_env(t):
    ph = (t % BEAT) / BEAT
    return np.exp(-ph * 6)


def palette(v, shift):
    r = 0.5 + 0.5 * np.cos(6.2831 * (v + shift + 0.00))
    g = 0.5 + 0.5 * np.cos(6.2831 * (v + shift + 0.33))
    b = 0.5 + 0.5 * np.cos(6.2831 * (v + shift + 0.67))
    return np.stack([r, g, b], -1)


def overlay_text(img, lines, t, glow=(255, 0, 200)):
    d = ImageDraw.Draw(img)
    for text, size, yy, t0 in lines:
        if t < t0:
            continue
        k = min(1, (t - t0) * 3)
        f = font(int(size * (0.6 + 0.4 * k + 0.08 * beat_env(t))))
        bw = d.textbbox((0, 0), text, font=f)
        tx = (W - (bw[2] - bw[0])) // 2
        jitter = int(4 * beat_env(t))
        for dx, dy, c in [(-jitter, 0, glow), (jitter, 0, (0, 255, 255))]:
            d.text((tx + dx, yy + dy), text, font=f, fill=c)
        d.text((tx, yy), text, font=f, fill=(255, 255, 255))
    return img


def frame_hyperspace(i):
    t = i / FPS
    b = beat_env(t)
    speed = 1.5 + 0.12 * t * t / DUR * 4
    depth = 1.0 / R + t * speed
    twist = A * 3 / np.pi + 0.3 * np.sin(t * 0.7) / R + t * 0.4
    rings = np.sin(depth * 6) * np.sin(twist * 6.2831)
    v = depth * 0.15 + rings * 0.25 + b * 0.2
    col = palette(v, t * 0.1)
    bright = np.clip(rings * 0.5 + 0.6, 0, 1) * np.clip(R * 1.6, 0, 1)
    core = np.exp(-R * (8 - 5 * b))[..., None]
    img = col ** 2.2 * bright[..., None] ** 1.5 * (0.9 + 0.6 * b) + core * np.array([1, 0.8, 1])
    # star streaks
    streak = (np.sin(A * 90 + np.floor(depth * 3)) > 0.995) * np.clip(R - 0.2, 0, 1)
    img += streak[..., None]
    img = np.clip(img, 0, 1)
    if 5.6 < t < 6.0:                       # flash drop
        img = 1 - img
    im = Image.fromarray((img * 255).astype(np.uint8))
    return overlay_text(im, [("ENTER", 70, 120, 0.5),
                             ("HYPERSPACE", 54, 205, 1.2),
                             ("no way back", 34, 820, 7.5)], t)


def frame_kaleido(i):
    t = i / FPS
    b = beat_env(t)
    seg = 6 + 2 * int(t // 3)               # symmetry increases every 3s
    a = np.abs(((A + t * 0.3) % (2 * np.pi / seg)) - np.pi / seg)
    zr = R * (1.2 + 0.3 * np.sin(t * 0.8)) * (1 - 0.15 * b)
    px, py = zr * np.cos(a), zr * np.sin(a)
    acc = np.zeros_like(px)
    for k in range(5):                      # folded fractal layers
        px, py = np.abs(px) - 0.55, np.abs(py) - 0.2 * np.sin(t * 0.5 + k)
        rot = t * 0.2 + k
        px, py = px * np.cos(rot) - py * np.sin(rot), px * np.sin(rot) + py * np.cos(rot)
        d = np.sqrt(px * px + py * py)
        acc += 0.02 / (np.abs(np.sin(d * 8 - t * 2)) + 0.03)
    col = palette(acc * 0.08 + R * 0.3, t * 0.15 + 0.5)
    img = col * np.clip(acc * 0.9, 0, 2.5)[..., None] ** 1.3 * (0.8 + 0.7 * b)
    img *= np.clip(1.3 - R * 0.55, 0, 1)[..., None]
    img = np.clip(img, 0, 1)
    im = Image.fromarray((img * 255).astype(np.uint8))
    return overlay_text(im, [("INFINITE", 72, 110, 0.6),
                             ("BLOOM", 96, 195, 1.4),
                             ("watch it again.", 34, 830, 8.0)], t,
                        glow=(255, 120, 0))


def make_audio(path, style):
    sr = 44100
    tt = np.arange(int(sr * DUR)) / sr
    ph = (tt % BEAT) / BEAT
    kick = np.sin(2 * np.pi * (50 + 120 * np.exp(-ph * 30)) * ph * BEAT) * np.exp(-ph * 8)
    hat = np.random.default_rng(1).standard_normal(len(tt)) * np.exp(-((tt + BEAT / 2) % BEAT) / BEAT * 40) * 0.15
    notes = [55, 65.4, 49, 73.4] if style == 1 else [61.7, 73.4, 55, 82.4]
    bar = (tt // (BEAT * 4)).astype(int) % 4
    f = np.array(notes)[bar]
    bass = np.sign(np.sin(2 * np.pi * f * tt)) * 0.18 * (1 - np.exp(-ph * 10))
    pad = sum(np.sin(2 * np.pi * f * m * tt + np.sin(tt * 0.5 + m)) for m in (2, 3, 4.01)) * 0.05
    riser = np.sin(2 * np.pi * (200 + 1800 * (tt / DUR) ** 3) * tt) * 0.04 * (tt / DUR)
    mix = kick * 0.6 + hat + bass + pad + riser
    mix *= np.clip(tt * 2, 0, 1) * np.clip((DUR - tt) * 2, 0, 1)
    mix = np.tanh(mix * 1.4) * 0.8
    data = (mix * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(data.tobytes())


def render(name, fn, style):
    wav = f"/tmp/{name}.wav"
    make_audio(wav, style)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", wav,
           "-vf", "scale=1080:1920:flags=lanczos", "-c:v", "libx264", "-preset", "fast",
           "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
           "-shortest", "-movflags", "+faststart", f"{name}.mp4"]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(N):
        p.stdin.write(fn(i).tobytes())
        if i % 60 == 0:
            print(name, i, "/", N, flush=True)
    p.stdin.close(); p.wait()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("1", "both"):
        render("reel1_hyperspace", frame_hyperspace, 1)
    if which in ("2", "both"):
        render("reel2_kaleido", frame_kaleido, 2)
