"""Answer typed questions about one state with a Plumb model."""

from plumber import Plumber

engine = Plumber("totum-labs/plumb-nemotron-3.5-lightning-30b-a3b")

state = {
    "subject": "Duplicate charge on invoice #4411",
    "body": "We were billed twice for March. Refund the duplicate today or we cancel our plan.",
}
questions = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, integrations",
            "sales": "pricing, upgrades, new accounts",
        },
    },
    "churn_risk": {"type": "noul", "instructions": "Does the customer threaten to leave?"},
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["not urgent", "soon", "blocking"],
    },
}

out = engine.decide(state, questions)
for qid, answer in out["answers"].items():
    print(qid, answer)
print(out["usage"], f"{out['latency_ms']} ms")
