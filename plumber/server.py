"""plumber serve --model <merged release or adapter dir> --port 8123   ->   POST /v1/systemone (TypeSafe request/response schema)."""
from __future__ import annotations
import argparse, json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .engine import Plumber, DEFAULT_BASE

def make_handler(engine):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _send(self, code, obj):
            body = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        def do_GET(self): self._send(200, {"ok": True, "model": engine.model_id})
        def do_POST(self):
            if not self.path.rstrip("/").endswith("/v1/systemone"): return self._send(404, {"error": "POST /v1/systemone"})
            try: req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0) or b"{}")
            except Exception as e: return self._send(400, {"error": f"bad json: {e}"})
            t0 = time.perf_counter()
            try: resp = engine.decide(req.get("state", ""), req.get("questions") or {})
            except Exception as e: return self._send(400, {"error": f"{type(e).__name__}: {str(e)[:300]}"})
            self._send(200, resp)
    return H

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True, help="merged release (HF repo id or dir) or adapter dir"); ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--port", type=int, default=8123); ap.add_argument("--dtype", default="bfloat16"); ap.add_argument("--max_len", type=int, default=32768)
    a = ap.parse_args(); eng = Plumber(a.model, base=a.base, dtype=a.dtype, max_len=a.max_len)
    print(f"[plumber] {eng.model_id} loaded in {eng.load_seconds:.0f}s on :{a.port}  POST /v1/systemone", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(eng)).serve_forever()

if __name__ == "__main__":
    main()
