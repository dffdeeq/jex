import json

from fastapi.testclient import TestClient

from jex.model import JexModel
from jex.server import create_app


def test_http_api(tiny_backbone, payload):
    model = JexModel(tiny_backbone)
    with TestClient(create_app(model)) as client:
        r = client.post("/v1/systemone", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert set(body["answers"]) == set(payload["questions"])
        assert body["answers"]["department"]["choice"] in {"billing", "technical", "sales"}
        assert 0.0 <= body["answers"]["churn_risk"]["noul"] <= 1.0

        r = client.post("/v1/systemone/stream", json=payload)
        names = [json.loads(line)["name"] for line in r.text.splitlines()]
        assert sorted(names) == sorted(payload["questions"])

        r = client.post("/v1/ask", json={"state": payload["state"], "name": "x",
                                         "question": {"type": "noul", "instructions": "Is it urgent?"}})
        assert r.status_code == 200 and r.json()["answer"]["type"] == "noul"

        bad = {"state": "s", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {"a": ""}}}}
        assert client.post("/v1/systemone", json=bad).status_code == 422
        assert client.post("/v1/systemone/stream", json=bad).status_code == 422
