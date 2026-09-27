"""Call a running `plumber serve` endpoint with the TypeSafe request shape."""

import json
import sys
import urllib.request

url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8123/v1/systemone"
body = {
    "state": "Invoice #4411 was billed twice for March. The customer asks for a refund today or they cancel.",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages"},
        },
        "churn_risk": {"type": "noul", "instructions": "Does the customer threaten to leave?"},
    },
}
req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
print(json.dumps(json.load(urllib.request.urlopen(req)), indent=2))
