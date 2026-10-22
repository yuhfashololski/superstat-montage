"""Where to point the vertical crop.

Game film is wide (16:9) and a TikTok is tall (9:16), so every frame needs a
crop that follows the action. This is a deliberately cheap stand-in: a
low-res motion-energy centroid, smoothed so the "camera" glides instead of
jittering.

In production this module is where Superstat's own player tracking plugs in -
follow the bounding box of the player whose highlight it is.
"""

from __future__ import annotations

import subprocess

import numpy as np

W, H, RATE = 192, 108, 10  # analysis resolution and sample rate


def _gray_frames(video: str, start: float, end: float) -> np.ndarray:
    cmd = [
        "ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
        "-i", video, "-vf", f"fps={RATE},scale={W}:{H},format=gray",
        "-f", "rawvideo", "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    n = len(raw) // (W * H)
    return np.frombuffer(raw[: n * W * H], np.uint8).reshape(n, H, W).astype(np.float32)


def _smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    if len(x) < 3:
        return x
    r = int(3 * sigma) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    padded = np.pad(x, r, mode="edge")
    return np.convolve(padded, k, mode="valid")


def focus_track(video: str, start: float, end: float, hint: float | None = None):
    """Return f(src_t) -> horizontal focus in 0..1 for the window [start, end]."""
    if hint is not None:
        return lambda t: hint
    frames = _gray_frames(video, start, end)
    if len(frames) < 2:
        return lambda t: 0.5
    diff = np.abs(np.diff(frames, axis=0))
    # Ignore the top band (scoreboard/crowd) and global brightness flicker.
    diff[:, : int(H * 0.25)] = 0
    diff -= np.median(diff, axis=(1, 2), keepdims=True)
    diff = np.clip(diff, 0, None)
    col = diff.sum(axis=1)  # (n, W) motion per column
    xs = np.arange(W) / (W - 1)
    weight = col.sum(axis=1)
    cx = np.where(weight > 1e-3, (col * xs).sum(axis=1) / np.maximum(weight, 1e-3), 0.5)
    # Pull weak-motion frames toward the centre, then smooth like a camera operator.
    conf = np.clip(weight / (np.percentile(weight, 75) + 1e-3), 0, 1)
    cx = 0.5 + (cx - 0.5) * conf
    cx = _smooth(cx, sigma=RATE * 0.6)
    times = start + (np.arange(len(cx)) + 1) / RATE
    return lambda t: float(np.interp(t, times, cx))
