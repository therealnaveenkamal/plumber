# Side-by-side demo: the normal model and the same model with its plumb, thinking on in both, one server, real time.
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
Type "Hey, I'm on call tonight for the payments service. Can you help me make quick calls if something breaks?"
Sleep 500ms
Enter
Wait+Screen@90s /NORMAL MODEL\s+\d+\.\d s\s+WITH PLUMB\s+\d+\.\d s/
Sleep 2s

Type "Alert just fired: p99 latency on checkout went from 180 ms to 2.4 s right after the 21:05 deploy, error rate is 1.8%, database and CPU look normal. What should I do? One line, please."
Sleep 500ms
Enter
Wait+Screen@120s /faster with the plumb/
Sleep 4s
