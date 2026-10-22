"""python -m montage --events game.json --player p1 --out reel.mp4 [--sound hellraiser.mp3]"""

import argparse
import json
from pathlib import Path

from . import template as tpl
from .edit import render
from .events import load_game


def main() -> None:
    ap = argparse.ArgumentParser(prog="montage", description="One-tap HellRaiser mixtape edit from Superstat game film.")
    ap.add_argument("--events", required=True, help="game events JSON (Superstat-style timeline export)")
    ap.add_argument("--player", required=True, help="player id to build the edit for")
    ap.add_argument("--out", default="mixtape.mp4")
    ap.add_argument("--sound", help="local copy of the HellRaiser sound for a preview with audio. "
                                   "Without it the edit is exported silent, ready to add the sound in TikTok.")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result = render(load_game(args.events), args.player, args.out, args.sound)
    print(json.dumps(result, indent=2))
    if not args.sound:
        print(f"\nPost it: add the sound '{tpl.SOUND_TITLE}' in TikTok and set its start to "
              f"0:{tpl.SOUND_START:05.2f} - every make lands on the beat.")


if __name__ == "__main__":
    main()
