"""Game event model.

The input mirrors what Superstat already produces for its video timeline: a list
of tagged events (made shots, rebounds, assists, steals, blocks, ...) with a
timestamp, the player involved and, for shots, a court location.

`t` is the moment the play "lands" - ball through the net, the block, the
steal. Everything else (clip windows, slow-mo, the "+3" pop, the beat the cut
lands on) is positioned relative to it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# type -> (points, pop text, label under the pop, highlight score)
EVENT_STYLE = {
    "3PT_MADE": (3, "+3", "THREE", 5.0),
    "2PT_MADE": (2, "+2", "BUCKET", 3.0),
    "FT_MADE": (1, "+1", "FREE THROW", 0.5),
    "BLK": (0, "BLOCK", "REJECTED", 5.0),
    "STL": (0, "STEAL", "PICKED", 4.0),
    "AST": (0, "+1 AST", "DIME", 2.5),
    "REB": (0, "+1 REB", "BOARD", 1.0),
    # Not a stat: a celebration / crowd / bench moment used as b-roll by the mixtape style.
    "MOMENT": (0, "", "", 0.0),
}

TAG_OVERRIDES = {
    # tag -> (pop text, label, bonus score)
    "dunk": (None, "SLAM", 3.0),
    "and1": (None, "AND-ONE", 2.0),
    "buzzer": (None, "AT THE BUZZER", 3.0),
    "layup": (None, "LAYUP", 0.0),
    "fastbreak": (None, "IN TRANSITION", 0.5),
    "deep": (None, "FROM DEEP", 1.0),
}


@dataclass
class Player:
    id: str
    name: str
    number: int | None = None
    team: str | None = None
    # Optional overrides - normally derived from events, but Superstat's own box
    # score (or a team's final score) can be passed straight through.
    final_line: dict | None = None
    hook: list[str] | None = None


@dataclass
class Event:
    id: str
    t: float
    type: str
    player_id: str
    period: str | None = None
    shot: dict | None = None  # half court: {"x": 0..1 across from left sideline, "y": 0..1 out from the baseline}
    tags: list[str] = field(default_factory=list)
    # Optional per-event crop hint, 0..1 across the frame. Superstat's player
    # tracking would supply this (or a bbox track) instead of our motion heuristic.
    focus_x: float | None = None
    # Optional: never start this clip before this source time (skip title cards, cutaways).
    clip_start: float | None = None
    clip_end: float | None = None
    # Optional vertical framing hint, 0..1 down the frame.
    focus_y: float | None = None
    # Virtual camera (what Superstat's player/ball tracking would supply): shooter position at
    # release and rim position, as [x, y] fractions of the frame, plus the release time.
    release: float | None = None
    shooter: list[float] | None = None
    rim: list[float] | None = None
    # MOMENT only: how many beats to hold it for, and playback speed.
    beats: int | None = None
    speed: float | None = None

    @property
    def points(self) -> int:
        return EVENT_STYLE.get(self.type, (0,))[0]

    @property
    def pop_text(self) -> str:
        return EVENT_STYLE[self.type][1]

    @property
    def label(self) -> str:
        label = EVENT_STYLE[self.type][2]
        for tag in self.tags:
            if tag in TAG_OVERRIDES and TAG_OVERRIDES[tag][1]:
                label = TAG_OVERRIDES[tag][1]
        return label

    @property
    def score(self) -> float:
        s = EVENT_STYLE.get(self.type, (0, "", "", 0.0))[3]
        return s + sum(TAG_OVERRIDES.get(tag, (None, None, 0.0))[2] for tag in self.tags)

    @property
    def is_shot(self) -> bool:
        return self.type.endswith("_MADE") or self.type.endswith("_MISS")


@dataclass
class Game:
    video: str
    title: str
    subtitle: str
    players: dict[str, Player]
    events: list[Event]
    credit: str = ""  # footage attribution, printed small on the end card
    # Clean region of the frame (fractions of height) - excludes burned-in graphics/watermarks.
    usable_top: float = 0.0
    usable_bottom: float = 1.0

    def for_player(self, player_id: str) -> list[Event]:
        return sorted((e for e in self.events if e.player_id == player_id), key=lambda e: e.t)


def load_game(path: str | Path) -> Game:
    path = Path(path)
    raw = json.loads(path.read_text())
    players = {p["id"]: Player(**p) for p in raw["players"]}
    events = [Event(**e) for e in raw["events"]]
    unknown = {e.type for e in events} - EVENT_STYLE.keys()
    if unknown:
        raise ValueError(f"unknown event types: {sorted(unknown)}")
    video = raw["game"]["video"]
    if not Path(video).is_absolute():
        video = str((path.parent / video).resolve())
    return Game(
        video=video,
        title=raw["game"].get("title", ""),
        subtitle=raw["game"].get("subtitle", ""),
        players=players,
        events=events,
        credit=raw["game"].get("credit", ""),
        usable_top=raw["game"].get("usable_top", 0.0),
        usable_bottom=raw["game"].get("usable_bottom", raw["game"].get("usable_height", 1.0)),
    )


def box_score(events: list[Event]) -> dict[str, int]:
    """Running stat line from the events shown (what the tally/end card display)."""
    line = {"PTS": 0, "REB": 0, "AST": 0, "STL": 0, "BLK": 0, "3PM": 0}
    for e in events:
        line["PTS"] += e.points
        if e.type == "3PT_MADE":
            line["3PM"] += 1
        elif e.type in ("REB", "AST", "STL", "BLK"):
            line[e.type] += 1
    return line
