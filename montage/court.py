"""Shot map, in the spirit of Superstat's Court View: a FIBA half court, basket at the top.

Shot locations are half-court fractions: x across the court (0 = left sideline),
y out from the baseline (0 = baseline, 1 = half-way line, 14m).
"""

from __future__ import annotations

import math
from functools import lru_cache

from PIL import Image, ImageDraw

from . import graphics as gfx

COURT_W_M, COURT_D_M = 15.0, 14.0
RIM = (7.5, 1.575)
SS = 3  # supersampling for clean thin lines


@lru_cache(maxsize=8)
def base(width: int, panel: bool = True) -> Image.Image:
    """Line-art half court (cached per size)."""
    m = 0.06 * width
    h = int((width - 2 * m) * COURT_D_M / COURT_W_M + 2 * m)
    W, H = width * SS, h * SS
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if panel:
        d.rounded_rectangle((0, 0, W - 1, H - 1), radius=int(0.08 * W), fill=(0, 0, 0, 150),
                            outline=(255, 255, 255, 40), width=SS)
    s = (W - 2 * m * SS) / COURT_W_M
    P = lambda x, y: (m * SS + x * s, m * SS + y * s)
    line = (255, 255, 255, 120)
    lw = max(1, int(1.2 * SS * width / 260))
    d.rectangle((*P(0, 0), *P(15, 14)), outline=line, width=lw)
    d.rectangle((*P(5.05, 0), *P(9.95, 5.8)), outline=line, width=lw)  # key
    cx, cy = P(7.5, 5.8)
    r = 1.8 * s
    d.arc((cx - r, cy - r, cx + r, cy + r), 0, 180, fill=line, width=lw)  # free-throw circle
    # three-point line: corners at 0.9m from the sidelines, arc radius 6.75m
    corner_y = RIM[1] + math.sqrt(6.75**2 - 6.6**2)
    d.line((*P(0.9, 0), *P(0.9, corner_y)), fill=line, width=lw)
    d.line((*P(14.1, 0), *P(14.1, corner_y)), fill=line, width=lw)
    a0 = math.degrees(math.atan2(corner_y - RIM[1], -6.6))
    a1 = math.degrees(math.atan2(corner_y - RIM[1], 6.6))
    rx, ry = P(*RIM)
    R = 6.75 * s
    d.arc((rx - R, ry - R, rx + R, ry + R), a1, a0, fill=line, width=lw)
    hx, hy = P(7.5, 14)
    d.arc((hx - r, hy - r, hx + r, hy + r), 180, 360, fill=line, width=lw)  # centre circle
    d.line((*P(6.6, 1.2), *P(8.4, 1.2)), fill=line, width=lw)  # backboard
    d.ellipse((rx - 0.23 * s, ry - 0.23 * s, rx + 0.23 * s, ry + 0.23 * s), outline=gfx.GREEN + (220,), width=lw)
    return img.resize((width, h), Image.LANCZOS)


def to_px(width: int, x: float, y: float) -> tuple[float, float]:
    m = 0.06 * width
    s = (width - 2 * m) / COURT_W_M
    return m + x * COURT_W_M * s, m + y * COURT_D_M * s


def rim_px(width: int) -> tuple[float, float]:
    return to_px(width, RIM[0] / COURT_W_M, RIM[1] / COURT_D_M)


def draw(width: int, shots: list[tuple[dict, float]], panel: bool = True, flight: float = 0.9) -> Image.Image:
    """Court with shots. `shots` = [(shot, age)] where age = seconds since that make landed
    (negative = still in the air: the ball's arc draws from the shot spot to the rim)."""
    img = base(width, panel).copy()
    W, H = img.size
    layer = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    rx, ry = (v * SS for v in rim_px(width))
    dot = 0.028 * width * SS
    for shot, age in shots:
        sx, sy = (v * SS for v in to_px(width, shot["x"], shot["y"]))
        if age < 0:  # ball in the air
            k = gfx.ease_out_cubic(1 + age / flight)
            if k <= 0:
                continue
            mx, my = (sx + rx) / 2, min(sy, ry) - 0.18 * width * SS
            pts = [((1 - u) ** 2 * sx + 2 * (1 - u) * u * mx + u * u * rx,
                    (1 - u) ** 2 * sy + 2 * (1 - u) * u * my + u * u * ry) for u in [i / 30 * k for i in range(31)]]
            d.line(pts, fill=gfx.GREEN + (170,), width=max(2, int(0.008 * width * SS)))
            d.ellipse((sx - dot * 0.7, sy - dot * 0.7, sx + dot * 0.7, sy + dot * 0.7), outline=gfx.GREEN + (220,),
                      width=max(2, int(0.006 * width * SS)))
            continue
        if age < 0.6:  # landing pulse
            k = gfx.ease_out_cubic(age / 0.6)
            rr = dot * (1 + 3 * k)
            d.ellipse((sx - rr, sy - rr, sx + rr, sy + rr), outline=gfx.GREEN + (int(230 * (1 - k)),),
                      width=max(2, int(0.008 * width * SS)))
        grow = gfx.ease_out_back(min(1.0, age / 0.18)) if age < 0.18 else 1.0
        r = dot * grow
        d.ellipse((sx - r, sy - r, sx + r, sy + r), fill=gfx.GREEN + (255,))
    layer = layer.resize((W, H), Image.LANCZOS)
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out.alpha_composite(gfx.glow(layer, max(2, width // 60), gfx.GREEN, 0.9))
    out.alpha_composite(img)
    out.alpha_composite(layer)
    return out
