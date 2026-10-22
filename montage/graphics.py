"""Type, brand marks and easing - drawn with Pillow."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Superstat brand, sampled from superstatsport.com
GREEN = (22, 249, 178)
DARK = (18, 30, 26)
DARKER = (10, 18, 15)
WHITE = (255, 255, 255)

CANVAS = (1080, 1920)

FONT_DIRS = [Path("/usr/share/fonts/opentype/inter"), Path(__file__).parent / "fonts"]
FONT_FALLBACKS = {
    "pop": ["InterDisplay-BlackItalic.otf", "Inter-BlackItalic.otf"],
    "black": ["InterDisplay-Black.otf", "Inter-Black.otf"],
    "bold": ["Inter-Bold.otf", "InterDisplay-Bold.otf"],
    "semibold": ["Inter-SemiBold.otf", "Inter-Medium.otf"],
    "light": ["InterDisplay-Light.otf", "Inter-Light.otf"],
    "extralight": ["InterDisplay-ExtraLight.otf", "Inter-ExtraLight.otf"],
}


@lru_cache(maxsize=None)
def font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    for d in FONT_DIRS:
        for name in FONT_FALLBACKS[kind]:
            if (d / name).exists():
                return ImageFont.truetype(str(d / name), size)
    return ImageFont.load_default(size)


def clamp(x: float, a: float = 0.0, b: float = 1.0) -> float:
    return max(a, min(b, x))


def ease_out_cubic(x: float) -> float:
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_out_back(x: float, k: float = 2.2) -> float:
    x = clamp(x)
    return 1 + (k + 1) * (x - 1) ** 3 + k * (x - 1) ** 2


def ramp(age: float, start: float, dur: float) -> float:
    return clamp((age - start) / dur)


def text_size(text: str, f: ImageFont.FreeTypeFont, tracking: int = 0) -> tuple[int, int]:
    l, t, r, b = f.getbbox(text)
    return r - l + tracking * max(0, len(text) - 1), b - t


def draw_text(img: Image.Image, xy, text, f, fill, anchor="la", tracking=0):
    d = ImageDraw.Draw(img)
    if not tracking:
        d.text(xy, text, font=f, fill=fill, anchor=anchor)
        return
    w, _ = text_size(text, f, tracking)
    x, y = xy
    if anchor[0] == "m":
        x -= w / 2
    elif anchor[0] == "r":
        x -= w
    for ch in text:
        d.text((x, y), ch, font=f, fill=fill, anchor="l" + anchor[1])
        x += f.getlength(ch) + tracking


def tracked(text: str, size: int, kind: str = "semibold", tracking: float = 0.5, fill=WHITE) -> Image.Image:
    """A single line of letter-spaced type on a transparent sprite."""
    f = font(kind, size)
    t = int(size * tracking)
    w, _ = text_size(text, f, t)
    img = Image.new("RGBA", (w + 40, int(size * 1.6)), (0, 0, 0, 0))
    draw_text(img, (img.width / 2, img.height / 2), text, f, fill, anchor="mm", tracking=t)
    return img


def glow(layer: Image.Image, radius: int, color=GREEN, strength: float = 1.0) -> Image.Image:
    alpha = layer.getchannel("A").filter(ImageFilter.GaussianBlur(radius))
    if strength != 1.0:
        alpha = alpha.point(lambda v: min(255, int(v * strength)))
    g = Image.new("RGBA", layer.size, color + (0,))
    g.putalpha(alpha)
    return g


def with_glow(sprite: Image.Image, radius: int, color, strength=1.0) -> Image.Image:
    pad = radius * 2
    big = Image.new("RGBA", (sprite.width + 2 * pad, sprite.height + 2 * pad), (0, 0, 0, 0))
    big.alpha_composite(sprite, (pad, pad))
    out = glow(big, radius, color, strength)
    out.alpha_composite(big)
    return out


def logo_mark(size: int, color=GREEN) -> Image.Image:
    """The two stacked parallelograms from the Superstat app icon."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    s = size / 1024
    top = [(150, 545), (400, 235), (820, 235), (565, 560)]
    bot = [(185, 795), (340, 590), (575, 590), (685, 470), (870, 470), (600, 795)]
    d.polygon([(x * s, y * s) for x, y in top], fill=color)
    d.polygon([(x * s, y * s) for x, y in bot], fill=color)
    return img


def wordmark(height: int, color=GREEN, with_mark=True) -> Image.Image:
    f = font("black", int(height * 1.05))
    w, _ = text_size("superstat", f)
    mark = height + 8 if with_mark else 0
    img = Image.new("RGBA", (w + mark + 10, int(height * 1.5)), (0, 0, 0, 0))
    if with_mark:
        img.alpha_composite(logo_mark(int(height * 1.3), color), (0, 0))
    d = ImageDraw.Draw(img)
    d.text((mark, height * 0.62), "superstat", font=f, fill=color, anchor="lm")
    return img


def fade(sprite: Image.Image, a: float) -> Image.Image:
    if a >= 0.999:
        return sprite
    s = sprite.copy()
    s.putalpha(s.getchannel("A").point(lambda v: int(v * a)))
    return s


def paste_center(canvas: Image.Image, sprite: Image.Image, cx: float, cy: float, alpha: float = 1.0):
    if alpha <= 0.01:
        return
    sprite = fade(sprite, alpha)
    x, y = int(cx - sprite.width / 2), int(cy - sprite.height / 2)
    canvas.alpha_composite(sprite, (max(0, x), max(0, y)), (max(0, -x), max(0, -y)))
