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
Type "I'm grading a maths quiz tonight. Can you help me check the answers?"
Sleep 500ms
Enter
Wait+Screen@90s /NORMAL MODEL\s+\d+\.\d s\s+WITH PLUMB\s+\d+\.\d s/
Sleep 2s

Type `Posts are placed every 8 metres along a straight 288-metre fence, with a post at both ends. How many posts are needed? A student wrote 37. What's the correct answer?\n- 37 posts\n- 38 posts\n- 36 posts\n- 2,304 posts`
Sleep 500ms
Enter
Wait+Screen@120s /faster with the plumb/
Sleep 4s
