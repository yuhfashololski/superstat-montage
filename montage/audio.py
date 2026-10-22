"""Audio helpers, numpy only: decode/encode, beat detection (used to map the
template's song), and synthesised impact/whoosh hits."""

from __future__ import annotations

import subprocess
import wave

import numpy as np

SR = 44100


def read_audio(path: str, sr: int = SR) -> np.ndarray:
    """Decode any audio file to mono float32 via ffmpeg."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, np.float32)


def write_wav(path: str, x: np.ndarray, sr: int = SR) -> None:
    x = np.asarray(x, np.float32)
    if x.ndim == 1:
        x = x[:, None]
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def detect_beats(path: str, sr: int = 22050) -> tuple[float, list[float]]:
    """Return (bpm, beat times in seconds) using spectral-flux onsets + autocorrelation."""
    x = read_audio(path, sr)
    hop, n_fft = 512, 2048
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop] * np.hanning(n_fft)
    mag = np.log1p(np.abs(np.fft.rfft(frames, axis=1)))
    flux = np.clip(np.diff(mag, axis=0), 0, None).sum(axis=1)
    flux = (flux - flux.mean()) / (flux.std() + 1e-9)
    fps = sr / hop

    # Tempo: strongest autocorrelation lag in 70-180 BPM, with a mild prior around 110.
    ac = np.correlate(flux, flux, mode="full")[len(flux) - 1:]
    lags = np.arange(len(ac))
    bpm_of = lambda lag: 60 * fps / np.maximum(lag, 1)
    valid = (bpm_of(lags) >= 70) & (bpm_of(lags) <= 180)
    prior = np.exp(-0.5 * (np.log2(bpm_of(lags) / 110) / 0.9) ** 2)
    lag = int(np.argmax(np.where(valid, ac * prior, -np.inf)))
    # Sub-frame peak (parabolic) - a 0.5% tempo error drifts a whole frame every few seconds.
    y0, y1, y2 = ac[lag - 1], ac[lag], ac[lag + 1]
    period = (lag + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-9)) / fps

    # Phase: offset whose grid collects the most onset energy.
    best, phase = -np.inf, 0.0
    for off in np.linspace(0, period, 64, endpoint=False):
        idx = np.round((off + np.arange(0, len(flux) / fps - off, period)) * fps).astype(int)
        idx = idx[idx < len(flux)]
        score = flux[idx].sum()
        if score > best:
            best, phase = score, off

    # Refine: snap each grid beat to its local onset peak, then fit a straight line.
    grid = np.arange(phase, len(flux) / fps, period)
    win = int(0.07 * fps)
    snapped = []
    for g in grid:
        i = int(round(g * fps))
        lo, hi = max(0, i - win), min(len(flux), i + win + 1)
        if hi - lo > 2 and flux[lo:hi].max() > 1.0:
            snapped.append((g, (lo + np.argmax(flux[lo:hi])) / fps))
    if len(snapped) > 8:
        n = np.round((np.array([s[0] for s in snapped]) - phase) / period)
        period, phase = np.polyfit(n, [s[1] for s in snapped], 1)

    # flux[i] is the change into frame i+1, centred half a window later
    shift = (hop + n_fft / 2) / sr
    beats = list(np.arange(phase % period, len(x) / sr, period) + shift)
    return 60 / period, [b for b in beats if b < len(x) / sr]


def _env(n: int, attack: float, decay: float, sr: int = SR) -> np.ndarray:
    t = np.arange(n) / sr
    return np.minimum(t / max(attack, 1e-4), 1) * np.exp(-t / decay)


def impact(sr: int = SR) -> np.ndarray:
    """Sub-boom + noise crack for the moment the '+3' slams in."""
    n = int(0.9 * sr)
    t = np.arange(n) / sr
    freq = 110 * np.exp(-t * 6) + 38
    boom = np.sin(2 * np.pi * np.cumsum(freq) / sr) * _env(n, 0.002, 0.28)
    crack = np.random.default_rng(1).standard_normal(n) * _env(n, 0.0005, 0.03)
    return 0.9 * boom + 0.35 * crack


def whoosh(dur: float = 0.45, sr: int = SR) -> np.ndarray:
    """Riser into a cut / slow-mo."""
    n = int(dur * sr)
    noise = np.random.default_rng(2).standard_normal(n)
    spec = np.fft.rfft(noise)
    f = np.fft.rfftfreq(n, 1 / sr)
    spec *= np.exp(-((f - 2500) / 2000) ** 2)
    shaped = np.fft.irfft(spec, n)
    env = np.sin(np.linspace(0, np.pi, n)) ** 2 * np.linspace(0.3, 1, n)
    return 0.6 * shaped / (np.abs(shaped).max() + 1e-9) * env


def sfx_track(duration: float, impacts: list[float], whooshes: list[float], sr: int = SR) -> np.ndarray:
    out = np.zeros(int(duration * sr) + sr)
    hit, sw = impact(sr), whoosh(sr=sr)
    for t in impacts:
        i = int(t * sr)
        out[i:i + len(hit)] += hit[: len(out) - i]
    for t in whooshes:
        i = max(0, int((t - 0.4) * sr))
        out[i:i + len(sw)] += sw[: len(out) - i]
    return out[: int(duration * sr)]
