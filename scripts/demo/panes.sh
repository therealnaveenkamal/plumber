#!/usr/bin/env bash
# Side-by-side demo on one server: left, the normal model; right, the same model with its plumb. Thinking is on in
# both panes (THINK=0 turns it off in both), and the normal model's prompt lists an unrelated tool so both prompts are
# shaped alike. Keystrokes go to both panes at once, so each message starts both runs together. The bar at the bottom
# compares the two turn by turn; SPEED=0 shows times only, and ANSWER="..." reveals the correct answer once both finish.
#
#   URL=http://localhost:8000 scripts/demo/panes.sh
set -euo pipefail
URL=${URL:-http://localhost:8000}
PY=${PY:-python}
HERE=$(cd "$(dirname "$0")" && pwd)
CHAT="$PY $HERE/../demo_chat.py --url $URL"
DIR=$(mktemp -d)
[ "${SPEED:-1}" = 1 ] || touch "$DIR/times_only"
[ -z "${ANSWER:-}" ] || printf '%s' "$ANSWER" > "$DIR/answer"
TOKENS=${MAX_TOKENS:-8192}
THINK_ARG=; [ "${THINK:-1}" = 1 ] && THINK_ARG=--think
export LANG=${LANG:-en_US.UTF-8} LC_ALL=${LC_ALL:-en_US.UTF-8}
T="tmux -u -L plumb-demo -f /dev/null"
$T kill-server 2> /dev/null || true
$T new-session -d -s demo -x "${COLS:-200}" -y "${ROWS:-50}" \
    "$CHAT --no-plumb $THINK_ARG --tool-prompt --max_tokens $TOKENS --report $DIR/normal.jsonl"
$T split-window -h -t demo "$CHAT $THINK_ARG --max_tokens $TOKENS --report $DIR/plumb.jsonl"
$T set -t demo pane-border-status top
$T set -t demo pane-border-lines heavy
$T set -t demo pane-border-format \
    "#{?#{==:#{pane_index},0},#[fg=colour203#,bold]  NORMAL MODEL${TITLE_SUFFIX:-}  ,#[fg=colour114#,bold]  WITH PLUMB${TITLE_SUFFIX:-}  }"
$T set -t demo status on
$T set -t demo status-position bottom
$T set -t demo status-interval 1
$T set -t demo status-style "bg=colour236,fg=colour252"
$T set -t demo status-format[0] "#[align=centre]#($PY $HERE/status.py $DIR)"
$T setw -t demo synchronize-panes on
exec $T attach -t demo
