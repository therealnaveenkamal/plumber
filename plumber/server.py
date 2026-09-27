"""plumber serve --model <merged release or adapter dir>   ->   POST /v1/systemone (the System One contract)."""

from __future__ import annotations

import argparse
import json
import os
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__
from .engine import Plumber


def make_handler(engine, api_key: str | None = None):
    """Routes: GET / (health), GET /v1/models, POST /v1/systemone. ``api_key`` requires ``Authorization: Bearer``.

    Requests are accepted concurrently but run through the model one at a time: the trunk's Triton kernels
    (autotuning on first use of a shape) are not safe to call from several threads at once.
    """
    infer = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("x-typesafe-request-id", uuid.uuid4().hex)
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            if not api_key or not self.path.startswith("/v1/"):
                return True
            return self.headers.get("Authorization", "") == f"Bearer {api_key}"

        def do_GET(self):
            if not self._authorized():
                return self._send(401, {"error": "invalid api key"})
            if self.path.rstrip("/") == "/v1/models":
                return self._send(
                    200,
                    {
                        "data": [
                            {
                                "name": engine.model_id,
                                "base": engine.base,
                                "engine": f"plumber {__version__}",
                            }
                        ]
                    },
                )
            self._send(200, {"ok": True, "model": engine.model_id})

        def do_POST(self):
            if not self._authorized():
                return self._send(401, {"error": "invalid api key"})
            if self.path.rstrip("/") != "/v1/systemone":
                return self._send(404, {"error": "POST /v1/systemone"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                req = json.loads(self.rfile.read(n) or b"{}")
            except (ValueError, TypeError) as e:
                return self._send(400, {"error": f"bad json: {e}"})
            if (
                not isinstance(req, dict)
                or not isinstance(req.get("questions"), dict)
                or not req["questions"]
            ):
                return self._send(
                    422, {"error": "request needs `state` and a non-empty `questions` object"}
                )
            try:
                with infer:
                    resp = engine.decide(req.get("state", ""), req["questions"])
            except ValueError as e:
                return self._send(422, {"error": str(e)[:300]})
            resp["model"] = req.get("model") or resp["model"]
            self._send(200, resp)

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", required=True, help="merged release (HF repo id or dir) or adapter dir"
    )
    ap.add_argument("--base", default=None, help="override the base recorded in the plumb")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to accept other machines")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max_len", type=int, default=32768)
    a = ap.parse_args()
    eng = Plumber(a.model, base=a.base, dtype=a.dtype, max_len=a.max_len)
    key = os.environ.get("PLUMBER_API_KEY") or None
    print(
        f"[plumber] {eng.model_id} loaded in {eng.load_seconds:.0f}s  http://{a.host}:{a.port}/v1/systemone"
        + ("  (bearer key required)" if key else ""),
        flush=True,
    )
    ThreadingHTTPServer((a.host, a.port), make_handler(eng, api_key=key)).serve_forever()


if __name__ == "__main__":
    main()
