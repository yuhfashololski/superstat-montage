"""Reading game film: probe + a sequential frame reader with sub-frame blending (for slow-mo)."""

from __future__ import annotations

import json
import subprocess

import numpy as np


def probe(video: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate:format=duration", "-of", "json", video],
        capture_output=True, check=True, text=True,
    ).stdout
    info = json.loads(out)
    s = info["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {"fps": float(num) / float(den), "w": int(s["width"]), "h": int(s["height"]),
            "duration": float(info["format"]["duration"])}


class SourceWindow:
    """Decode [start, end] once, then serve frames for monotonically increasing timestamps.
    Fractional positions are blended between neighbouring frames, which keeps slow-mo smooth."""

    def __init__(self, video: str, start: float, end: float, info: dict):
        self.start, self.fps = start, info["fps"]
        self.w, self.h = info["w"], info["h"]
        self.proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{end - start + 0.3:.3f}", "-i", video,
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        self.idx, self.cur, self.nxt = -1, None, None
        self._advance()
        self._advance()

    def _read(self):
        n = self.w * self.h * 3
        buf = self.proc.stdout.read(n)
        if len(buf) < n:
            return None
        return np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)

    def _advance(self):
        f = self._read()
        self.cur = self.nxt if self.nxt is not None else f
        self.nxt = f if f is not None else self.cur
        self.idx += 1

    def frame_at(self, t: float) -> np.ndarray:
        pos = max(0.0, (t - self.start) * self.fps)
        while self.idx < int(pos):
            self._advance()
        frac = pos - int(pos)
        if frac < 0.12 or self.nxt is self.cur:
            return self.cur
        if frac > 0.88:
            return self.nxt
        return (self.cur * (1 - frac) + self.nxt * frac).astype(np.uint8)

    def close(self):
        self.proc.stdout.close()
        self.proc.kill()
        self.proc.wait()
