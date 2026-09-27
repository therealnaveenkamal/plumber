"""The official TypeSafe SDK must round-trip against `plumber serve` unchanged: same path, auth, request and answer schemas."""

import threading
from http.server import ThreadingHTTPServer

import pytest
import torch

from plumber.core import PlumbModel
from plumber.engine import Plumber
from plumber.server import make_handler
from tests.test_core import Tok, tiny

typesafe_sdk = pytest.importorskip("typesafe_sdk")


def _tiny_engine() -> Plumber:
    eng = Plumber.__new__(Plumber)
    eng.tok, eng.pad, eng.max_len, eng.model_id = Tok(), 0, 4096, "plumb-tiny"
    eng.model = PlumbModel(tiny().eval().model, 64, d_proj=16).eval()
    eng.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return eng


def test_typesafe_sdk_round_trip():
    from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(_tiny_engine(), api_key="local"))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        client = TypeSafeClient(api_key="local", base_url=base_url, model="plumb-tiny")
        response = client.system_one(
            state="I was charged twice. Please fix this ASAP.",
            questions={
                "billing": Noul(instructions="Is this ticket about billing?"),
                "tone": Choice(
                    instructions="What is the customer's tone?",
                    criteria={"calm": None, "frustrated": None, "angry": None},
                ),
                "urgency": Score(
                    instructions="How urgent is this ticket?",
                    criteria=["can wait", "this week", "today"],
                ),
            },
        )
        assert 0.0 <= response.nouls["billing"].noul <= 1.0
        assert response.choices["tone"].choice in {"calm", "frustrated", "angry"}
        assert abs(sum(response.choices["tone"].probabilities.values()) - 1.0) < 1e-4
        assert 0.0 <= response.scores["urgency"].score <= 2.0
        assert {int(k) for k in response.scores["urgency"].legend} == {0, 1, 2}
        assert response.usage.output_tokens == 0 and response.usage.input_tokens > 0
        assert response.model == "plumb-tiny"
        # the SDK sent `Authorization: Bearer local`; a wrong key is refused
        bad = TypeSafeClient(api_key="nope", base_url=base_url, model="plumb-tiny")
        with pytest.raises(Exception, match=r"401|[Ii]nvalid|[Uu]nauthori"):
            bad.system_one(state="x", questions={"q": Noul(instructions="y?")})
    finally:
        server.shutdown()
