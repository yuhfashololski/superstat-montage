"""Renders the HellRaiser mixtape edit.

Look (matched to the top #mixtape hoops edits):
  * 1:1 picture, letterboxed in black, film grain, vignette
  * full colour on the build-up -> washed black & white (only reds survive) from the first hit
  * every effect is driven by the song map in template.py: flash, RGB split, shake and
    zoom punch on the hits, harder on the drop; a zoom pulse on every beat after the drop
  * speed ramps are smooth (monotone cubic) but each make still lands exactly on its beat
Superstat layer: the only saturated colour on screen is Superstat green - the stats.
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFilter

from . import audio, court, graphics as gfx, template as tpl
from .events import Event, Game, box_score
from .source import SourceWindow, probe
from .track import focus_track

W, H = gfx.CANVAS
BAND = 1080
BAND_Y = (H - BAND) // 2
FPS = 30
P = tpl.BEAT
DROP_T = tpl.DROP_BEAT * P
COLOR_UNTIL = 4 * P  # first hit

# hit kind -> (flash, shake px, rgb split px, zoom punch, decay seconds)
HIT_FX = {"hit": (0.35, 9, 7, 0.07, 0.22), "drop": (0.9, 28, 24, 0.22, 0.45), "808": (0.28, 15, 11, 0.10, 0.28)}

# --------------------------------------------------------------- grading ----

_GRAIN = [np.random.default_rng(i).normal(0, 8, (BAND, BAND, 1)).astype(np.float32) for i in range(8)]
_yy, _xx = np.mgrid[0:BAND, 0:BAND] / (BAND - 1) - 0.5
VIGNETTE = (1 - 0.6 * np.clip((np.sqrt(_xx**2 + _yy**2) - 0.3) / 0.42, 0, 1) ** 1.5)[..., None].astype(np.float32)


def grade_color(f: np.ndarray) -> np.ndarray:
    lum = f.mean(axis=2, keepdims=True)
    f = lum + (f - lum) * 1.2
    return (f - 0.5) * 1.1 + 0.5


def grade_mono(f: np.ndarray, pre_drop: bool) -> np.ndarray:
    """Washed mixtape B&W: reds kept, S-curve, faded blacks, cool shadows."""
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    # Keep reds and the orange of the ball (hue < ~24deg, saturated); wood floors (~30deg) go grey.
    mx, mn = np.maximum(np.maximum(r, g), b), np.minimum(np.minimum(r, g), b)
    sat = (mx - mn) / (mx + 1e-4)
    hue = 60 * (g - b) / (mx - mn + 1e-4)
    is_warm = (r >= mx - 1e-6) & (hue > -15)
    keep_mask = is_warm * np.clip((26 - hue) / 8, 0, 1) * np.clip((sat - 0.3) / 0.2, 0, 1)
    keep = (0.06 + 0.75 * keep_mask)[..., None]
    out = lum[..., None] + (f - lum[..., None]) * keep
    s = 0.5 - 0.5 * np.cos(np.pi * np.clip(out, 0, 1))
    out = (0.5 if pre_drop else 0.7) * s + (0.5 if pre_drop else 0.3) * out
    out = 0.05 + 0.9 * out
    if pre_drop:
        out *= 0.86  # moodier before the drop
    out[..., 2] += 0.035 * (1 - out[..., 1])
    return out


def bloom(f: np.ndarray, amount: float) -> np.ndarray:
    small = Image.fromarray((np.clip(f, 0, 1) * 255).astype(np.uint8)).resize((270, 270), Image.BILINEAR)
    hi = np.asarray(small.filter(ImageFilter.GaussianBlur(6)).resize((BAND, BAND), Image.BILINEAR)).astype(np.float32) / 255
    return f + amount * np.clip(hi - 0.55, 0, 1)


def rgb_split(f: np.ndarray, px: int) -> np.ndarray:
    if px < 1:
        return f
    out = f.copy()
    out[..., 0] = np.roll(f[..., 0], -px, axis=1)
    out[..., 2] = np.roll(f[..., 2], px, axis=1)
    return out


# ----------------------------------------------------------------- camera ----

def _smooth(x: float) -> float:
    x = gfx.clamp(x)
    return x * x * (3 - 2 * x)


def camera(e: Event, src_t: float, replay: bool = False):
    """Virtual camera: tight on the shooter at release, glide with the ball to the rim, ease out.
    Returns (x, y, zoom, weight) in frame fractions - weight blends with the default framing -
    or None without tracking data."""
    if e.rim and not (e.shooter and e.release is not None):
        # Rim only (broadcast footage with cuts): settle on the net just before the make,
        # stay on it through the swish, then let go.
        rx, ry = e.rim
        if replay:
            return rx, ry + 0.03, 2.0, 1.0
        w = _smooth((src_t - (e.t - 0.55)) / 0.4) * (1 - _smooth((src_t - (e.t + 0.6)) / 0.5))
        return rx, ry + 0.06, 1.3, w
    if not (e.shooter and e.rim and e.release is not None):
        return None
    (sx, sy), (rx, ry) = e.shooter, e.rim
    if replay:  # locked tight on the net, creeping in
        k = _smooth((src_t - (e.t - 0.4)) / 0.6)
        return rx, ry + 0.03, 2.0 + 0.25 * k, 1.0
    z_tight, z_rim = 1.55, 1.3
    if src_t <= e.release - 0.15:
        w = _smooth((src_t - (e.release - 1.2)) / 1.05)  # ease in from the wide shot during the possession
        return sx, sy, z_tight, w
    if src_t <= e.t:
        k = _smooth((src_t - (e.release - 0.15)) / (e.t - e.release + 0.15))
        zoom = z_tight + (z_rim - z_tight) * k
        return sx + (rx - sx) * k, sy + (ry - sy) * k, zoom, 1.0
    k = _smooth((src_t - e.t) / 1.0)  # after the make: drift back to show the reaction
    return rx + (0.5 - rx) * 0.4 * k, ry + (sy - ry) * 0.35 * k, z_rim - 0.15 * k, 1.0


def crop_square(frame: np.ndarray, game: Game, e: Event, fx: float, zoom: float, shake, cam=None) -> Image.Image:
    fh, fw = frame.shape[:2]
    top, bot = game.usable_top * fh, game.usable_bottom * fh
    cy = e.focus_y * fh if e.focus_y is not None else (top + bot) / 2
    if cam:
        x, y, cz, w = cam
        fx = fx + (x - fx) * w
        cy = cy + (y * fh - cy) * w
        zoom *= 1 + (cz - 1) * w
    cy += shake[1]
    side = min(fw, bot - top) / zoom
    cy = float(np.clip(cy, top + side / 2, bot - side / 2))
    cx = float(np.clip(fx * fw + shake[0], side / 2, fw - side / 2))
    x0, y0 = int(cx - side / 2), int(cy - side / 2)
    region = frame[y0:y0 + int(side), x0:x0 + int(side)]
    return Image.fromarray(region).resize((BAND, BAND), Image.BICUBIC)


def hit_state(t: float):
    """Flash, shake, RGB split and zoom punch at time t, from the song map."""
    flash = shake_amp = split = punch = 0.0
    for beat, kind in tpl.HITS.items():
        age = t - beat * P
        fl, sh, rg, zp, dec = HIT_FX[kind]
        if 0 <= age < dec * 2:
            k = math.exp(-age / (dec * 0.35))
            flash = max(flash, fl * math.exp(-age / 0.06))
            shake_amp = max(shake_amp, sh * k)
            split = max(split, rg * k)
            punch = max(punch, zp * math.exp(-age / dec))
    return flash, shake_amp, split, punch


# ------------------------------------------------------------------- type ----

_cache: dict = {}


def sprite(key, make):
    if key not in _cache:
        _cache[key] = make()
    return _cache[key]


def draw_name_tag(canvas: Image.Image, name: str, hook: str, t: float):
    """Big thin name on the first hit (editor-tag style) in the top bar, stat hook underneath in green."""
    age = t - 4 * P
    dur = 4 * P
    if age < 0 or age > dur:
        return
    flicker = 1.0 if age > 0.1 else (1.0 if int(age * FPS) % 2 == 0 else 0.2)
    a = flicker * (1 - gfx.ramp(age, dur - 0.2, 0.2))
    tag = sprite(("tag", name), lambda: gfx.with_glow(gfx.tracked(name.upper(), 104, "light", 0.32), 16, gfx.WHITE, 0.5))
    s = 1.05 - 0.05 * gfx.ease_out_cubic(age / dur)
    tw = min(int(tag.width * s), W - 30)
    tag = tag.resize((tw, int(tag.height * tw / tag.width)), Image.BILINEAR)
    gfx.paste_center(canvas, tag, W / 2, BAND_Y - 170, a)
    if hook:
        sub = sprite(("hook", hook), lambda: gfx.with_glow(gfx.tracked(hook, 54, "bold", 0.4, gfx.GREEN), 12, gfx.GREEN, 0.9))
        sa = gfx.ease_out_cubic(gfx.ramp(age, 0.25, 0.25)) * (1 - gfx.ramp(age, dur - 0.2, 0.2))
        gfx.paste_center(canvas, sub, W / 2, BAND_Y - 75, sa)


CHIP_W = 250
CHIP_X, CHIP_Y = W - 40 - CHIP_W, BAND_Y + 40
FOOTER_H = 92


def flight_time(shot) -> float:
    """How long the ball is 'in the air' on the shot map: the slow-mo leading into the make."""
    return max(0.55, (shot.slot.land - shot.slot.start - shot.slot.real) * P)


@lru_cache(maxsize=2)
def _chip_panel(h: int) -> Image.Image:
    ss = 3
    img = Image.new("RGBA", (CHIP_W * ss, h * ss), (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle((0, 0, CHIP_W * ss - 1, h * ss - 1), radius=22 * ss, fill=(0, 0, 0, 165),
                                          outline=(255, 255, 255, 38), width=ss)
    return img.resize((CHIP_W, h), Image.LANCZOS)


def draw_stat_chip(canvas: Image.Image, shots: list[tuple[dict, float, float]], pops: list[tuple[Event, float]]):
    """Top-right chip: Superstat-style shot map + a footer that flashes '+3 THREE' on each make.
    On the map each make draws its arc to the rim during the slow-mo, then leaves a green dot."""
    live = [(sh, age, fl) for sh, age, fl in shots if age > -fl]
    if not live:
        return
    a = gfx.ease_out_cubic(gfx.clamp((live[0][1] + live[0][2]) / 0.25))
    court_img = court.draw(CHIP_W, [(sh, age if age >= 0 else age / fl * 0.9) for sh, age, fl in live], panel=False)
    chip = _chip_panel(court_img.height + FOOTER_H).copy()
    chip.alpha_composite(court_img, (0, 0))
    fy = court_img.height + FOOTER_H // 2 - 8
    active = [(e, age) for e, age in pops if 0 <= age < 2.2 * P]
    if active:
        e, age = active[-1]
        pop = sprite(("chip-pop", e.pop_text), lambda: gfx.with_glow(gfx.tracked(e.pop_text, 66, "pop", 0.0, gfx.GREEN), 10, gfx.GREEN, 1.2))
        lab = sprite(("chip-lab", e.label), lambda: gfx.tracked(e.label, 19, "bold", 0.3))
        k = gfx.ease_out_cubic(age / 0.1)
        out = gfx.ramp(age, 2.2 * P - 0.15, 0.15)
        pa = 1.0 if out == 0 else (1 - out) * (int(age * FPS) % 2 == 0)
        # sprites carry padding: tracked() 20px per side, with_glow() 2*radius per side
        pop_vis = pop.width - 80
        scale = 1.35 - 0.35 * k
        sp = pop.resize((int(pop.width * scale), int(pop.height * scale)), Image.BILINEAR)
        gfx.paste_center(chip, sp, 18 + pop_vis / 2, fy, pa)
        lab_x = 18 + pop_vis + 14
        room = CHIP_W - 16 - lab_x
        if lab.width - 40 > room:
            lab = lab.resize((int(lab.width * room / (lab.width - 40)), int(lab.height * room / (lab.width - 40))), Image.LANCZOS)
        lab_vis = lab.width * (1 - 40 / max(lab.width, 41))
        gfx.paste_center(chip, lab, lab_x + lab_vis / 2, fy + 6, gfx.ease_out_cubic(gfx.ramp(age, 0.06, 0.15)) * pa)
        flash = 1 - gfx.ease_out_cubic(age / 0.25)
        if flash > 0:  # the chip outline lights up green on impact
            ring = Image.new("RGBA", chip.size, (0, 0, 0, 0))
            ImageDraw.Draw(ring).rounded_rectangle((1, 1, CHIP_W - 2, chip.height - 2), radius=22, outline=gfx.GREEN + (int(230 * flash),), width=3)
            chip.alpha_composite(ring)
    else:
        idle = sprite("chip-idle", lambda: gfx.tracked("SHOT MAP", 20, "semibold", 0.5, gfx.WHITE + (150,)))
        gfx.paste_center(chip, idle, CHIP_W / 2, fy + 2)
    canvas.alpha_composite(gfx.fade(chip, a), (CHIP_X, CHIP_Y))


def draw_watermark(canvas: Image.Image, alpha: float):
    if alpha <= 0:
        return
    wm = sprite("wm", lambda: _tracked_with(38))
    gfx.paste_center(canvas, wm, W / 2, BAND_Y + BAND + 90, alpha * 0.85)


def _tracked_with(h: int) -> Image.Image:
    label = gfx.tracked("TRACKED WITH", int(h * 0.62), "semibold", 0.35, gfx.WHITE + (200,))
    mark = gfx.wordmark(h)
    img = Image.new("RGBA", (label.width + mark.width + 10, max(label.height, mark.height)), (0, 0, 0, 0))
    img.alpha_composite(label, (0, (img.height - label.height) // 2))
    img.alpha_composite(mark, (label.width + 10, (img.height - mark.height) // 2))
    return img


def end_card(still: np.ndarray, name: str, line: dict, shots: list[dict], age: float, credit: str) -> Image.Image:
    """Freeze on the last make, then the final line, the shot chart and the Superstat sign-off, on the beats."""
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    k = gfx.ease_out_cubic(gfx.ramp(age, 0, 0.5))
    bg = Image.fromarray(still).filter(ImageFilter.GaussianBlur(1 + 9 * k))
    bg = Image.blend(bg.convert("RGBA"), Image.new("RGBA", bg.size, (0, 0, 0, 255)), 0.65 * k)
    canvas.alpha_composite(bg, (0, BAND_Y))
    if age < 0.12:
        canvas = Image.blend(canvas, Image.new("RGBA", canvas.size, (255, 255, 255, 255)), 0.6 * (1 - age / 0.12))

    # player: "7 THREES" / "31 PTS"; team (final score passed in): "85-68"
    if line.get("3PM"):
        key, headline = "3PM", f"{line['3PM']} THREES"
    elif "PTS" in line:
        key, headline = "PTS", f"{line['PTS']} PTS"
    else:
        key, headline = None, "-".join(str(v) for v in line.values())
    rest = " · ".join(f"{v} {k}" for k, v in line.items() if key and k != key and v)
    pulse = 1 + 0.03 * math.exp(-((age % P) / 0.08))
    a1 = gfx.ease_out_cubic(gfx.ramp(age, 0.0, 0.2))
    nm = sprite(("end-name", name), lambda: gfx.tracked(name.upper(), 54, "light", 0.35))
    gfx.paste_center(canvas, nm, W / 2, BAND_Y + 120, a1)
    hl = sprite(("end-hl", headline), lambda: gfx.with_glow(gfx.tracked(headline, 140, "pop", 0.0, gfx.GREEN), 26, gfx.GREEN, 1.3))
    s = (1.35 - 0.35 * gfx.ease_out_cubic(gfx.ramp(age, 0.0, 0.12))) * pulse
    gfx.paste_center(canvas, hl.resize((int(hl.width * s), int(hl.height * s)), Image.BILINEAR), W / 2, BAND_Y + 255, a1)
    if rest:
        rs = sprite(("end-rest", rest), lambda: gfx.tracked(rest, 36, "semibold", 0.3))
        gfx.paste_center(canvas, rs, W / 2, BAND_Y + 365, gfx.ease_out_cubic(gfx.ramp(age, 2 * P, 0.2)))
    if shots:  # the shot chart fills in dot by dot on the beats
        a2 = gfx.ease_out_cubic(gfx.ramp(age, 2 * P, 0.25))
        chart = court.draw(430, [(sh, age - (2 + i) * P) for i, sh in enumerate(shots) if age - (2 + i) * P >= 0])
        canvas.alpha_composite(gfx.fade(chart, a2), (int(W / 2 - chart.width / 2), BAND_Y + 420))
    a3 = gfx.ease_out_cubic(gfx.ramp(age, 5 * P, 0.25))
    gfx.paste_center(canvas, sprite("end-wm", lambda: _tracked_with(46)), W / 2, BAND_Y + 1000, a3)
    if credit:
        cr = sprite(("credit", credit), lambda: gfx.tracked(credit, 20, "semibold", 0.05, gfx.WHITE + (110,)))
        gfx.paste_center(canvas, cr, W / 2, H - 140, a3)
    return canvas


# ----------------------------------------------------------------- render ----

def render(game: Game, player_id: str, out: str, sound: str | None = None, log=print) -> dict:
    player = game.players[player_id]
    events = game.for_player(player_id)
    makes = [e for e in events if e.type.endswith("_MADE") and e.type != "FT_MADE" and "miss" not in e.tags]
    shots = tpl.fill(makes)
    line = player.final_line or box_score(events)
    hook = " · ".join(f"{v} {k}" for k, v in list(line.items())[:2]) if line else ""
    if line.get("3PM"):
        hook = f"{line['3PM']} THREES"
    info = probe(game.video)
    end_b = tpl.end_beat(shots)
    total = (end_b + tpl.END_BEATS) * P
    log(f"HellRaiser template: {len({id(x.event) for x in shots})} makes, {len(shots)} shots, {total:.1f}s")

    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-"]
    if sound:
        sr = audio.SR
        a = audio.read_audio(sound, sr)[int(tpl.SOUND_START * sr): int((tpl.SOUND_START + total) * sr)].copy()
        n = int(0.9 * sr)
        a[-n:] *= np.linspace(1, 0, n)
        wav = str(Path(out).with_suffix(".wav"))
        audio.write_wav(wav, a)
        cmd += ["-i", wav, "-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", "20", "-maxrate", "10M", "-bufsize", "20M",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    frame_no = 0
    still = None
    mapped = [(sh.event.shot, sh.slot.land * P, flight_time(sh)) for sh in shots if sh.event.shot and not sh.slot.replay]
    for si, shot in enumerate(shots):
        e = shot.event
        t0 = shot.slot.start * P
        n = int(round(shot.slot.end * P * FPS)) - int(round(t0 * FPS))
        track = focus_track(game.video, shot.src_start, shot.src_end, e.focus_x)
        src = SourceWindow(game.video, shot.src_start, shot.src_end, info)
        land_t = shot.slot.land * P
        big = shot.slot.land == tpl.DROP_BEAT
        for fi in range(n):
            t = frame_no / FPS
            age = t - t0
            src_t = shot.remap(age)
            frame = src.frame_at(src_t)
            flash, shake_amp, split, punch = hit_state(t)
            post = t >= DROP_T
            zoom = (1.0 + 0.06 * t / DROP_T) if not post else 1.12 * (1 + 0.035 * math.exp(-((t - DROP_T) % P) / 0.07))
            zoom *= 1 + punch
            cut_k = 1 - gfx.ease_out_cubic(age / 0.16) if (si > 0 and age < 0.16) else 0.0
            if post and cut_k:
                zoom *= 1 + 0.3 * cut_k
            shake = (shake_amp * math.sin(t * 97), shake_amp * math.cos(t * 71))
            band = crop_square(frame, game, e, track(src_t), zoom, shake, camera(e, src_t, shot.slot.replay))
            if post and cut_k > 0.15:  # whip into the cut
                band = band.rotate((6 if si % 2 else -6) * cut_k, resample=Image.BILINEAR).filter(ImageFilter.BoxBlur(int(8 * cut_k)))
            f = np.asarray(band).astype(np.float32) / 255.0
            if t < COLOR_UNTIL:
                f = grade_color(f)
            else:
                f = grade_mono(f, pre_drop=not post)
                if not post:
                    f = bloom(f, 0.5)
            f = rgb_split(f, int(split))
            f = f * VIGNETTE + _GRAIN[frame_no % len(_GRAIN)] / 255.0
            if flash:
                f = f * (1 - flash) + flash
            if not post and cut_k:  # dip from black between pre-drop shots
                f *= 1 - cut_k
            arr = (np.clip(f, 0, 1) * 255).astype(np.uint8)
            still = arr

            canvas = Image.new("RGBA", (W, H), (0, 0, 0, 255))
            canvas.alpha_composite(Image.fromarray(arr).convert("RGBA"), (0, BAND_Y))
            draw_name_tag(canvas, player.name, hook, t)
            draw_stat_chip(canvas, [(sh, t - lt, fl) for sh, lt, fl in mapped], [(x.event, t - x.slot.land * P) for x in shots if not x.slot.replay])
            draw_watermark(canvas, gfx.ramp(t, 4 * P + 0.3, 0.4))
            enc.stdin.write(canvas.convert("RGB").tobytes())
            frame_no += 1
        src.close()
        log(f"  slot {si + 1}: {'replay of ' if shot.slot.replay else ''}{e.type} @ {e.t:.2f}s -> lands beat {shot.slot.land:g}{' (DROP)' if big else ''}")

    end_t0 = end_b * P
    while frame_no < int(round(total * FPS)):
        t = frame_no / FPS
        enc.stdin.write(end_card(still, player.name, line, [m[0] for m in mapped], t - end_t0, game.credit).convert("RGB").tobytes())
        frame_no += 1
    enc.stdin.close()
    enc.wait()
    return {"out": out, "duration": round(total, 2), "makes": len({id(x.event) for x in shots}), "sound": tpl.SOUND_TITLE,
            "sound_start": f"{tpl.SOUND_START:.2f}s", "with_audio": bool(sound)}
