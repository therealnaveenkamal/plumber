"""The `plumbify` command line: every command is listed, and `plumbify chat` starts light (no torch, transformers or
vLLM) with thinking on by default."""

import subprocess
import sys

from plumbify.chat import parser
from plumbify.cli import main


def test_commands_are_listed(capsys):
    assert main(["--help"]) == 0
    out = capsys.readouterr().out
    for cmd in ("train", "package", "train-head", "eval-head", "chat"):
        assert f"  {cmd} " in out


def test_chat_imports_nothing_heavy():
    code = "import sys, plumbify.chat; print(sorted(m for m in ('torch', 'transformers', 'vllm') if m in sys.modules))"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert out.strip() == "[]"


def test_chat_thinks_by_default():
    assert parser().parse_args([]).think is True
    assert parser().parse_args(["--no-think"]).think is False
    assert parser().parse_args([]).max_tokens == 8192
