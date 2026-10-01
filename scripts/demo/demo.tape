# Side-by-side demo: the normal model (thinking on) and the same model with its plumb, one server, real time.
#   vhs scripts/demo/demo.tape && scripts/demo/make_gif.sh
# Needs vhs, tmux and ffmpeg, and a plumbed model served on localhost:8000.
Output "docs/demo.mp4"
Require tmux

Set Shell "bash"
Set FontSize 18
Set Width 2000
Set Height 1150
Set Padding 14
Set Framerate 12
Set TypingSpeed 26ms
Set Theme "Catppuccin Mocha"

Hide
Type `PY=.venv/bin/python COLS=150 ROWS=40 scripts/demo/panes.sh` Enter
Sleep 3s
Show

Sleep 1.5s
Type "Hi! I'm covering the support inbox today. Can you help me triage a few tickets?"
Sleep 500ms
Enter
Wait+Screen@90s /NORMAL MODEL\s+\d+\.\d s\s+WITH PLUMB\s+\d+\.\d s/
Sleep 2s

Type "First one: 'I was charged twice for invoice 4411 and now the app keeps logging me out. Fix it today or I cancel.' Which team should take it? One line, please."
Sleep 500ms
Enter
Wait+Screen@120s /faster with the plumb/
Sleep 4s
