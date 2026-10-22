#!/usr/bin/env bash
# Demo footage: "Hawaii Basketball - Highlights From Northern Arizona Game"
# by Warrior Insider (https://vimeo.com/301560697), CC BY 3.0, via Wikimedia Commons.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p footage
curl -fL -A "superstat-montage-demo/0.1" -o footage/hawaii.webm \
  "https://upload.wikimedia.org/wikipedia/commons/transcoded/a/ac/Hawaii_Basketball_-_Highlights_From_Northern_Arizona_Game.webm/Hawaii_Basketball_-_Highlights_From_Northern_Arizona_Game.webm.720p.vp9.webm"
