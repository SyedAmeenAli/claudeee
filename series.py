"""THE LAST SIGNAL - a 3-episode apocalypse mini-series for Instagram Reels.

Each episode: narrated voice-over (Kokoro TTS), painted procedural illustrations
with slow camera moves, falling ash, burned-in captions, drone score, cliffhanger.

Usage: python3 series.py <episode 1|2|3>
Needs: numpy, Pillow, kokoro-onnx, soundfile, ffmpeg, and KOKORO_DIR with
k.onnx + voices.bin.
"""
import os, sys, subprocess, textwrap
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H, FPS = 540, 960, 24
SR = 24000
KD = os.environ.get("KOKORO_DIR", ".")

# (scene_type, sentence)
EPISODES = {
1: ("RED TUESDAY", [
 ("city", "The sky turned red on a Tuesday."),
 ("city", "At first, we thought it was smoke. Then the birds stopped singing. All of them. At the same time."),
 ("space", "At 3:14 in the afternoon, every screen on Earth went black. Every phone. Every radio. Every heartbeat monitor."),
 ("space", "And then, a sound. A single tone, coming from somewhere above the clouds."),
 ("dark", "By nightfall, the cities were silent. No lights. No sirens. Just that tone, humming through the walls."),
 ("room", "My name is Maya. I fixed radios for a living. So when everyone else ran, I did the only thing I knew how to do."),
 ("room", "I listened."),
 ("room", "Under the tone, there was a voice. Faint. Broken. Repeating the same words."),
 ("eye", "It said: Maya. Don't trust the sky. You have six days."),
 ("eye", "It was my voice. And it was coming from the future."),
]),
2: ("THE SILENT CITIES", [
 ("road", "Day two. I left the city at dawn, following the signal north."),
 ("city", "The highways were full of cars. Engines still running. Doors wide open. But no one inside."),
 ("shadow", "Where people had been standing, there were shadows burned into the concrete. Thousands of them. Frozen mid-step."),
 ("road", "On the fourth day, I found others. Twelve survivors, hiding in an old train tunnel."),
 ("fire", "A boy named Eli told me the tone was not a signal. It was a countdown. Getting faster every night."),
 ("space", "And far above us, something enormous was moving behind the red clouds. Blocking out the stars, one by one."),
 ("fire", "That night, we heard footsteps outside the tunnel. Slow. Hundreds of them."),
 ("shadow", "The missing people had come back. Every single one of them."),
 ("eye", "They were smiling. They were calling our names."),
 ("eye", "And none of them were breathing."),
]),
3: ("ZERO HOUR", [
 ("road", "Day six. Only Eli and I made it out of the tunnel."),
 ("tower", "The signal led us to a radio tower on the edge of the world. Rusted. Ancient. Still humming."),
 ("tower", "Inside, someone had covered the walls with notes. Dates. Warnings. Maps. All in my handwriting."),
 ("room", "There was a transmitter on the table, already tuned. And a note that said: Tell her everything. Make it count."),
 ("space", "Outside, the sky split open. The thing behind the clouds was not a ship. It was an eye. And it was looking at us."),
 ("room", "Eli grabbed my arm. He said: Maya, if you send this message, it starts all over again."),
 ("eye", "I understood then. I was never warning myself. I was the one who called it here."),
 ("room", "The countdown hit zero. I pressed the button. And I spoke."),
 ("city", "Maya. Don't trust the sky. You have six days."),
 ("city", "Somewhere, far away, the sky turned red. On a Tuesday."),
]),
}

PAL = {  # sky top, sky bottom, glow color
 "city": ((40, 5, 10), (230, 70, 20), (255, 150, 60)),
 "space": ((5, 0, 15), (120, 10, 40), (255, 60, 80)),
 "dark": ((2, 2, 8), (40, 20, 40), (180, 60, 60)),
 "room": ((15, 8, 5), (70, 35, 15), (255, 170, 80)),
 "eye": ((10, 0, 0), (90, 0, 10), (255, 40, 40)),
 "road": ((30, 10, 20), (200, 90, 50), (255, 190, 120)),
 "shadow": ((25, 20, 25), (130, 100, 90), (240, 200, 160)),
 "fire": ((8, 4, 4), (50, 20, 10), (255, 120, 30)),
 "tower": ((20, 0, 25), (180, 40, 60), (255, 90, 120)),
}


def font(size):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def serif(size):
    p = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
    return ImageFont.truetype(p, size) if os.path.exists(p) else font(size)


def grad(w, h, top, bot):
    t = np.linspace(0, 1, h)[:, None, None] ** 0.8
    return (np.array(top) * (1 - t) + np.array(bot) * t) * np.ones((1, w, 1))


def skyline(d, w, h, base, rng, col, ruined=True, n=40):
    x = -20
    while x < w + 20:
        bw = rng.integers(20, 70)
        bh = rng.integers(int(h * 0.12), int(h * 0.42))
        top = base - bh
        pts = [(x, base), (x, top)]
        if ruined and rng.random() < 0.6:
            for k in range(3):
                pts.append((x + bw * (k + 1) / 4, top + rng.integers(-15, 40)))
        pts += [(x + bw, top + rng.integers(0, 30)), (x + bw, base)]
        d.polygon(pts, fill=col)
        x += bw + rng.integers(-5, 10)


def paint(kind, seed):
    """Paint a 1.3x oversize illustration for a scene."""
    rng = np.random.default_rng(seed)
    w, h = int(W * 1.3), int(H * 1.3)
    top, bot, glow = PAL[kind]
    arr = grad(w, h, top, bot)
    # glow / sun
    yy, xx = np.mgrid[0:h, 0:w]
    gx, gy = w * rng.uniform(0.3, 0.7), h * rng.uniform(0.35, 0.55)
    r = np.sqrt((xx - gx) ** 2 + (yy - gy) ** 2)
    arr += np.array(glow) * np.exp(-r / (w * 0.22))[..., None] * 0.8
    # cloud bands
    for k in range(6):
        cy = rng.uniform(0.05, 0.6) * h
        band = np.exp(-((yy - cy - 25 * np.sin(xx / rng.uniform(40, 120) + k)) / rng.uniform(10, 40)) ** 2)
        arr += np.array(glow) * band[..., None] * 0.18
    # stars for space / dark
    if kind in ("space", "dark", "tower"):
        s = rng.random((h, w)) > 0.9985
        arr[s] = 255
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    base = int(h * 0.78)
    far = tuple(int(c * 0.35) for c in bot)
    if kind == "space":
        # giant shape behind clouds
        d.ellipse([w * 0.1, h * 0.05, w * 0.9, h * 0.45], fill=tuple(int(c * 0.3) for c in top))
        for k in range(8):
            ang = rng.uniform(0, 6.28)
            d.line([(w / 2, h * 0.25), (w / 2 + 600 * np.cos(ang), h * 0.25 + 600 * np.sin(ang))],
                   fill=glow, width=1)
        skyline(d, w, h, base, rng, (5, 0, 5))
    elif kind == "eye":
        cx, cy = w / 2, h * 0.38
        d.ellipse([cx - 300, cy - 120, cx + 300, cy + 120], fill=(40, 0, 0))
        d.ellipse([cx - 120, cy - 110, cx + 120, cy + 110], fill=(200, 20, 20))
        d.ellipse([cx - 45, cy - 100, cx + 45, cy + 100], fill=(0, 0, 0))
        d.ellipse([cx - 20, cy - 60, cx - 2, cy - 30], fill=(255, 220, 200))
        skyline(d, w, h, base + 40, rng, (0, 0, 0))
    elif kind == "room":
        d.rectangle([0, 0, w, h], fill=(12, 8, 6))
        d.rectangle([w * 0.55, h * 0.15, w * 0.95, h * 0.45], fill=(150, 30, 20))  # window w/ red sky
        d.line([(w * 0.75, h * 0.15), (w * 0.75, h * 0.45)], fill=(12, 8, 6), width=8)
        d.line([(w * 0.55, h * 0.3), (w * 0.95, h * 0.3)], fill=(12, 8, 6), width=8)
        d.rectangle([0, h * 0.62, w, h], fill=(30, 18, 10))  # desk
        d.rectangle([w * 0.18, h * 0.5, w * 0.62, h * 0.63], fill=(55, 45, 35))  # radio
        for k in range(5):
            d.ellipse([w * (0.22 + k * 0.07), h * 0.53, w * (0.26 + k * 0.07), h * 0.56], fill=(255, 160, 60))
        d.ellipse([w * 0.42, h * 0.38, w * 0.62, h * 0.66], fill=(5, 3, 3))  # silhouette head
        d.rectangle([w * 0.36, h * 0.6, w * 0.7, h], fill=(5, 3, 3))
    elif kind == "road":
        skyline(d, w, h, int(h * 0.6), rng, far, n=30)
        d.polygon([(w * 0.47, h * 0.6), (w * 0.53, h * 0.6), (w * 1.1, h), (-w * 0.1, h)], fill=(25, 15, 15))
        d.rectangle([0, h * 0.6, w, h * 0.605], fill=far)
        for k in range(6):
            y0 = h * (0.62 + k * 0.06)
            d.rectangle([w * 0.495, y0, w * 0.505, y0 + 12 + k * 4], fill=(200, 170, 90))
        for k in range(4):  # abandoned cars
            cx = w * rng.uniform(0.1, 0.9); cy = h * rng.uniform(0.7, 0.9)
            d.rounded_rectangle([cx - 40, cy - 18, cx + 40, cy + 10], 8, fill=(10, 8, 8))
        d.ellipse([w * 0.47, h * 0.78, w * 0.53, h * 0.8], fill=(0, 0, 0))
        d.rectangle([w * 0.488, h * 0.69, w * 0.512, h * 0.79], fill=(0, 0, 0))  # lone walker
        d.ellipse([w * 0.485, h * 0.665, w * 0.515, h * 0.695], fill=(0, 0, 0))
    elif kind == "shadow":
        skyline(d, w, h, int(h * 0.55), rng, far)
        d.rectangle([0, h * 0.55, w, h], fill=(110, 95, 85))
        for k in range(40):  # burned shadows of people
            cx, cy = w * rng.uniform(0, 1), h * rng.uniform(0.6, 1)
            s = 0.5 + (cy / h - 0.55) * 2
            d.ellipse([cx - 6 * s, cy - 40 * s, cx + 6 * s, cy - 28 * s], fill=(20, 15, 15))
            d.rectangle([cx - 7 * s, cy - 28 * s, cx + 7 * s, cy], fill=(20, 15, 15))
    elif kind == "fire":
        d.rectangle([0, 0, w, h], fill=(6, 4, 4))
        d.ellipse([-w * 0.3, -h * 0.1, w * 1.3, h * 1.1], fill=(18, 10, 8))  # tunnel arch
        d.ellipse([w * 0.15, h * 0.25, w * 0.85, h * 1.0], fill=(4, 2, 2))
        for k in range(7):  # survivors round a fire
            cx = w * (0.2 + k * 0.1); cy = h * 0.82
            d.ellipse([cx - 14, cy - 70, cx + 14, cy - 42], fill=(0, 0, 0))
            d.rectangle([cx - 18, cy - 44, cx + 18, cy + 10], fill=(0, 0, 0))
    elif kind == "tower":
        d.rectangle([0, h * 0.8, w, h], fill=(10, 0, 10))
        tx = w / 2
        d.polygon([(tx - 70, h * 0.8), (tx + 70, h * 0.8), (tx + 8, h * 0.12), (tx - 8, h * 0.12)], fill=(8, 0, 8))
        for k in range(12):
            y0 = h * (0.8 - k * 0.055)
            ww = 70 * (1 - k / 13)
            d.line([(tx - ww, y0), (tx + ww, y0 - h * 0.05)], fill=(30, 5, 20), width=3)
            d.line([(tx + ww, y0), (tx - ww, y0 - h * 0.05)], fill=(30, 5, 20), width=3)
        d.ellipse([tx - 10, h * 0.1, tx + 10, h * 0.12], fill=(255, 40, 40))
    else:  # city / dark
        skyline(d, w, h, int(h * 0.68), rng, far)
        skyline(d, w, h, base, rng, tuple(int(c * 0.12) for c in bot))
        d.rectangle([0, base, w, h], fill=(5, 2, 2))
        if kind == "city":
            for k in range(60):
                x0, y0 = rng.uniform(0, w), rng.uniform(h * 0.5, base)
                d.rectangle([x0, y0, x0 + 3, y0 + 4], fill=(255, 140, 50) if rng.random() < 0.3 else (20, 8, 8))
    img = img.filter(ImageFilter.GaussianBlur(0.8))
    # film grain + vignette
    a = np.asarray(img).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    vig = 1 - 0.65 * (((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / 2
    a = a * np.clip(vig, 0.2, 1)[..., None] + rng.normal(0, 6, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def caption(text):
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    f = font(30)
    lines = textwrap.wrap(text, 26)
    y = 640
    for ln in lines:
        bw = d.textbbox((0, 0), ln, font=f)[2]
        d.text(((W - bw) // 2, y), ln, font=f, fill=(255, 255, 255),
               stroke_width=4, stroke_fill=(0, 0, 0))
        y += 40
    return lay


def title_card(ep, name):
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    for txt, f, y, c in [("THE LAST SIGNAL", serif(42), 200, (255, 60, 50)),
                         (f"EPISODE {ep}", font(26), 280, (255, 255, 255)),
                         (name, serif(36), 320, (255, 220, 200))]:
        bw = d.textbbox((0, 0), txt, font=f)[2]
        d.text(((W - bw) // 2, y), txt, font=f, fill=c, stroke_width=3, stroke_fill=(0, 0, 0))
    return lay


def end_card(ep):
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    d = ImageDraw.Draw(lay)
    msg = "TO BE CONTINUED..." if ep < 3 else "THE END...?"
    sub = f"EPISODE {ep + 1} — follow so you don't miss it" if ep < 3 else "or is it? watch Episode 1 again."
    for txt, f, y, c in [(msg, serif(44), 400, (255, 40, 40)), (sub, font(20), 480, (220, 220, 220))]:
        bw = d.textbbox((0, 0), txt, font=f)[2]
        d.text(((W - bw) // 2, y), txt, font=f, fill=c)
    return lay


def voice(lines, out):
    from kokoro_onnx import Kokoro
    k = Kokoro(os.path.join(KD, "k.onnx"), os.path.join(KD, "voices.bin"))
    chunks, spans, t = [np.zeros(int(SR * 4.0))], [], 4.0  # 3s title intro
    for _, s in lines:
        a, sr = k.create(s, voice="am_onyx", speed=0.8, lang="en-us")
        if sr != SR:
            raise SystemExit("unexpected sample rate")
        spans.append((t, t + len(a) / SR))
        chunks += [a, np.zeros(int(SR * 1.5))]
        t += len(a) / SR + 1.5
    chunks.append(np.zeros(int(SR * 5.0)))  # end card
    v = np.concatenate(chunks).astype(np.float32)
    sf.write(out, v, SR)
    return spans, len(v) / SR


def score(dur, out, ep):
    sr = 44100
    t = np.arange(int(sr * dur)) / sr
    root = [55.0, 51.9, 49.0][ep - 1]
    drone = sum(np.sin(2 * np.pi * root * m * t + 0.3 * np.sin(t * 0.2 * m)) / m for m in (1, 1.5, 2, 3.01))
    swell = 0.5 + 0.5 * np.sin(2 * np.pi * t / 16)
    beat = 60 / (50 + 40 * t / dur)  # heartbeat speeds up
    ph = (t % 1.0) * 0  # placeholder
    hb_t = np.cumsum(np.ones_like(t) / sr / np.interp(t, [0, dur], [1.2, 0.55]))
    ph = hb_t % 1.0
    heart = np.sin(2 * np.pi * 45 * t) * (np.exp(-ph * 25) + 0.7 * np.exp(-np.abs(ph - 0.18) * 25))
    noise = np.random.default_rng(ep).standard_normal(len(t))
    wind = np.convolve(noise, np.ones(200) / 200, "same") * 0.6
    mix = drone * 0.08 * (0.6 + 0.4 * swell) + heart * 0.25 + wind * 0.3
    mix *= np.clip(t / 2, 0, 1) * np.clip((dur - t) / 1.5, 0, 1)
    sf.write(out, np.tanh(mix).astype(np.float32), sr)


def render(ep):
    name, lines = EPISODES[ep]
    tag = f"ep{ep}"
    vwav, mwav = f"/tmp/{tag}_v.wav", f"/tmp/{tag}_m.wav"
    spans, dur = voice(lines, vwav)
    print(tag, "voice", round(dur, 1), "s", flush=True)
    score(dur, mwav, ep)
    arts = [paint(k, ep * 100 + i) for i, (k, _) in enumerate(lines)]
    caps = [caption(s) for _, s in lines]
    tcard, ecard = title_card(ep, name), end_card(ep)
    rng = np.random.default_rng(ep)
    ash = rng.random((140, 3))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", vwav, "-i", mwav,
           "-filter_complex", "[1:a]volume=1.6[v];[2:a]volume=0.7[m];[v][m]amix=inputs=2:duration=first:normalize=0[a]",
           "-map", "0:v", "-map", "[a]", "-vf", "scale=1080:1920:flags=lanczos",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
           f"the_last_signal_ep{ep}.mp4"]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    nf = int(dur * FPS)
    for i in range(nf):
        t = i / FPS
        idx = 0
        for j, (a, b) in enumerate(spans):
            if t >= a - 0.35:
                idx = j
        a0 = spans[idx][0] - 0.35
        b0 = spans[idx + 1][0] - 0.35 if idx + 1 < len(spans) else dur
        u = np.clip((t - a0) / max(b0 - a0, 0.1), 0, 1)
        art = arts[idx]
        aw, ah = art.size
        z = 1.0 + 0.18 * u if idx % 2 == 0 else 1.18 - 0.18 * u     # Ken Burns
        cw, ch = aw / (1.3 * z) * 1.0, ah / (1.3 * z)
        ox = (aw - cw) * (0.3 + 0.4 * u if idx % 3 else 0.7 - 0.4 * u)
        oy = (ah - ch) * 0.5
        fr = art.resize((W, H), Image.BILINEAR, box=(ox, oy, ox + cw, oy + ch))
        # crossfade from previous art in first 0.5s of a scene
        if idx > 0 and t - a0 < 0.5:
            prev = arts[idx - 1].resize((W, H), Image.BILINEAR,
                                        box=(0, 0, arts[idx - 1].size[0] / 1.3, arts[idx - 1].size[1] / 1.3))
            fr = Image.blend(prev, fr, (t - a0) / 0.5)
        d = ImageDraw.Draw(fr)
        for (ax, ay, az) in ash:  # falling ash / embers
            px = (ax * W + 30 * np.sin(t * (1 + az) + ax * 9)) % W
            py = (ay * H + t * (20 + 60 * az)) % H
            c = (255, 120 + int(80 * az), 60) if az > 0.8 else (180, 170, 160)
            d.ellipse([px, py, px + 1 + 2 * az, py + 1 + 2 * az], fill=c)
        sa, sb = spans[idx]
        if sa - 0.2 <= t <= sb + 0.5:
            fr.paste(caps[idx], (0, 0), caps[idx])
        if t < 3.2:
            fr.paste(tcard, (0, 0), tcard)
        if t > dur - 4.0:
            k = min(1, (t - (dur - 4.0)) / 0.6)
            fr = Image.blend(fr, ecard.convert("RGB"), k)
            fr.paste(ecard, (0, 0), ecard) if k >= 1 else None
        p.stdin.write(fr.tobytes())
        if i % 240 == 0:
            print(tag, i, "/", nf, flush=True)
    p.stdin.close(); p.wait()
    print(tag, "done", flush=True)


if __name__ == "__main__":
    render(int(sys.argv[1]))
