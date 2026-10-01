#!/usr/bin/env bash
# docs/demo.mp4 -> docs/demo.gif, small enough for a README (about 10 MB): 1100 px wide, 4 fps, and a 64-colour
# palette built from the whole video so the red and green labels keep their colours.
set -euo pipefail
IN=${1:-docs/demo.mp4}
OUT=${2:-docs/demo.gif}
ffmpeg -v error -y -i "$IN" -vf "fps=4,scale=1100:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64:stats_mode=full[p];[b][p]paletteuse=dither=none:diff_mode=rectangle" "$OUT"
