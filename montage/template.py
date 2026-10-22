"""The HellRaiser cut sheet.

One song, one structure, done properly. Mapped from the TikTok sound
"HellRaiser" by SlimeGetEm (the sound behind the top #mixtape hoops edits):

  * 140.0 BPM, beat grid phase 0.295s into the sound
  * edits use the sound from 0:11.87 (same start as the reference edits)
  * from there, in beats (1 beat = 0.4285s):
        b4   first hit      -> colour drops to black & white, name tag
        b12  second hit
        b20  THE DROP       -> best make lands here
        b23, b27, b31, b35  808 hits after the drop
  * a make is placed so the ball goes through the net exactly on its beat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .events import Event

BPM = 140.015
BEAT = 60.0 / BPM
SOUND_PHASE = 0.295
SOUND_START = SOUND_PHASE + 27 * BEAT  # 11.865s: edit beat 0 == sound beat 27
SOUND_TITLE = "HellRaiser - SlimeGetEm"

HITS = {4: "hit", 12: "hit", 20: "drop", 23: "808", 27: "808", 31: "808", 35: "808"}
DROP_BEAT = 20


@dataclass(frozen=True)
class Slot:
    start: float  # beat the shot starts on
    land: float  # beat the make lands on
    end: float  # beat the shot cuts on
    real: float  # beats of real-time build-up at the start
    hang: float  # speed of the slow-mo leading into the make
    after: float  # speed of the slow-mo after the make
    replay: bool = False  # tight slow-mo replay of the previous make (no new stat)


# Up to five makes. Slot 3 lands on the drop and gets the long slow-mo "hang".
SLOTS = [
    Slot(0, 4, 8, 3.0, 0.50, 0.45),
    Slot(8, 12, 14, 3.0, 0.50, 0.45),
    Slot(14, 20, 23, 2.0, 0.30, 0.40),
    Slot(23, 27, 29, 3.0, 0.50, 0.45),
    Slot(29, 31, 35, 1.5, 0.50, 0.45),
]
END_BEATS = 8  # end card, from the last slot's cut


def _pchip(xk: np.ndarray, yk: np.ndarray):
    """Monotone cubic interpolation (Fritsch-Carlson): smooth speed ramps that still
    hit every keyframe exactly, so the make lands on the beat."""
    h = np.diff(xk)
    d = np.diff(yk) / h
    m = np.zeros_like(yk)
    for i in range(1, len(xk) - 1):
        if d[i - 1] * d[i] > 0:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    m[0], m[-1] = d[0], d[-1]

    def f(x: float) -> float:
        i = int(np.clip(np.searchsorted(xk, x) - 1, 0, len(h) - 1))
        t = (x - xk[i]) / h[i]
        h00, h10, h01, h11 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t, -2 * t**3 + 3 * t**2, t**3 - t**2
        return float(h00 * yk[i] + h10 * h[i] * m[i] + h01 * yk[i + 1] + h11 * h[i] * m[i + 1])

    return f


@dataclass
class Shot:
    slot: Slot
    event: Event
    src_start: float
    src_end: float
    remap: object  # out-time (s, within shot) -> source time (s)

    @property
    def beats(self) -> float:
        return self.slot.end - self.slot.start


# Two makes: the second gets the whole possession as its build-up into the drop,
# then a tight slow-mo replay of that swish lands on the b27 808.
TWO_MAKE_SLOTS = [
    SLOTS[0],
    Slot(8, 20, 23, 7.0, 0.30, 0.40),
    Slot(23, 27, 31, 0.0, 0.22, 0.12, replay=True),
]


def fill(makes: list[Event]) -> list[Shot]:
    """Assign makes to slots in game order; the highest-scoring make takes the drop."""
    makes = sorted(makes, key=lambda e: e.t)[: len(SLOTS)]
    if len(makes) < 2:
        raise SystemExit("the HellRaiser template needs at least 2 made shots")
    if len(makes) == 2:
        return [_shot(s, e) for s, e in zip(TWO_MAKE_SLOTS, [makes[0], makes[1], makes[1]])]
    # 4 makes: the last one lands on the b27 808 and the end card hits on b31.
    slots = SLOTS if len(makes) == 5 else SLOTS[:3] + ([Slot(23, 27, 31, 3.0, 0.50, 0.45)] if len(makes) == 4 else [])
    drop_i = min(2, len(makes) - 1)
    best = max(range(len(makes)), key=lambda i: (makes[i].score, -abs(i - drop_i)))
    order = list(makes)
    order.insert(drop_i, order.pop(best))
    return [_shot(slot, e) for slot, e in zip(slots, order)]


def _shot(slot: Slot, e: Event) -> Shot:
    p = BEAT
    out_k = np.array([0.0, slot.real, slot.land - slot.start, slot.end - slot.start]) * p
    slow_src = (slot.land - slot.start - slot.real) * p * slot.hang
    src_k = np.array([e.t - slow_src - slot.real * p, e.t - slow_src, e.t, e.t + (slot.end - slot.land) * p * slot.after])
    if slot.real == 0:  # no real-time part: drop the duplicate keyframe
        out_k, src_k = np.delete(out_k, 1), np.delete(src_k, 1)
    lo = e.clip_start if e.clip_start is not None else 0.0
    hi = e.clip_end if e.clip_end is not None else 1e9
    if src_k[0] < lo:  # not enough build-up footage: squeeze the real-time part
        src_k[0] = lo
    if src_k[-1] > hi:  # stay inside the clip
        src_k[-1] = hi
    return Shot(slot, e, float(src_k[0]), float(src_k[-1]), _pchip(out_k, src_k))


def end_beat(shots: list[Shot]) -> float:
    return shots[-1].slot.end
