"""HTTP client for a running ``plumber serve`` endpoint (or any TypeSafe-compatible ``/v1/systemone`` server).

from plumber.client import Client
client = Client("http://127.0.0.1:8123")
client.decide(state, questions)            # same response shape as Plumber.decide
client.choice(state, "Which team?", {"billing": "...", "sales": "..."})
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class Client:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8123",
        api_key: str | None = None,
        timeout_s: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.headers = {"content-type": "application/json"}
        if api_key:
            self.headers["authorization"] = f"Bearer {api_key}"
        self.timeout_s = timeout_s

    def decide(
        self, state: Any, questions: dict[str, dict], model: str | None = None
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"state": state, "questions": questions}
        if model:
            body["model"] = model
        req = urllib.request.Request(
            f"{self.base_url}/v1/systemone", json.dumps(body).encode(), self.headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{e.code}: {e.read().decode(errors='replace')[:300]}") from e

    def choice(self, state: Any, instructions: str, criteria: dict[str, str] | list[str]) -> dict:
        return self.decide(
            state, {"q": {"type": "choice", "instructions": instructions, "criteria": criteria}}
        )["answers"]["q"]

    def noul(self, state: Any, instructions: str, criteria: dict | None = None) -> dict:
        return self.decide(
            state, {"q": {"type": "noul", "instructions": instructions, "criteria": criteria}}
        )["answers"]["q"]

    def score(self, state: Any, instructions: str, levels: list[str]) -> dict:
        return self.decide(
            state, {"q": {"type": "score", "instructions": instructions, "criteria": list(levels)}}
        )["answers"]["q"]

    def health(self) -> dict:
        with urllib.request.urlopen(f"{self.base_url}/", timeout=10) as resp:
            return json.loads(resp.read().decode())
