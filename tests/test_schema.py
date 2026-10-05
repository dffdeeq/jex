import math

import pytest

from jex.schema import Answer, SchemaError, build_response, parse_request


def test_parse_types(payload):
    req = parse_request(payload)
    by_name = {q.name: q for q in req.questions}
    assert by_name["department"].option_ids == ("billing", "technical", "sales")
    assert by_name["churn_risk"].option_ids == ("true", "false")
    assert by_name["urgency"].option_ids == ("0", "1", "2", "3")
    assert by_name["urgency"].option_descriptions[3] == "critical"


@pytest.mark.parametrize(
    "spec",
    [
        {"type": "bool", "instructions": "x"},
        {"type": "noul"},
        {"type": "noul", "instructions": ""},
        {"type": "choice", "instructions": "x", "criteria": {"a": ""}},
        {"type": "choice", "instructions": "x", "criteria": {str(i): "" for i in range(256)}},
        {"type": "choice", "instructions": "x", "criteria": "a,b"},
        {"type": "score", "instructions": "x", "criteria": ["only one"]},
        {"type": "score", "instructions": "x", "criteria": [str(i) for i in range(11)]},
        {"type": "score", "instructions": "x", "criteria": {"a": "b"}},
    ],
)
def test_invalid_questions(spec):
    with pytest.raises(SchemaError):
        parse_request({"state": "s", "questions": {"q": spec}})


def test_invalid_requests():
    for bad in [None, {}, {"state": "s"}, {"state": "s", "questions": {}}, {"questions": {"q": {}}}]:
        with pytest.raises(SchemaError):
            parse_request(bad)


def test_response_shape(payload):
    req = parse_request(payload)
    probs = {"department": [0.7, 0.2, 0.1], "churn_risk": [0.8, 0.2], "urgency": [0.0, 0.1, 0.9, 0.0]}
    answers = [Answer(q, probs[q.name]) for q in req.questions]
    resp = build_response("jex-test", answers, input_tokens=42)
    a = resp["answers"]
    assert a["department"]["choice"] == "billing"
    assert abs(a["department"]["confidence"] - 0.55) < 1e-9  # Jev: (K*p_max - 1) / (K - 1)
    assert a["churn_risk"] == {"type": "noul", "noul": 0.8}
    assert math.isclose(a["urgency"]["score"], 1.9)
    assert a["urgency"]["legend"]["2"] == "urgent"
    assert resp["usage"] == {"input_tokens": 42, "output_tokens": 0}


def test_response_rejects_invalid_distribution(payload):
    req = parse_request(payload)
    with pytest.raises(AssertionError):
        build_response("x", [Answer(req.questions[0], [0.5, 0.1, 0.1])], 0)


def test_structured_instructions_and_descriptions():
    """Jev accepts JSON objects as instructions and option descriptions."""
    req = parse_request({"state": {"clause": "x"}, "questions": {
        "unfair": {"type": "noul", "instructions": {"clause_type": {"name": "Liability"}, "question": "Is it?"}},
        "pick": {"type": "choice", "instructions": "Which?", "criteria": {"1": {"text": "first"}, "2": None}},
    }})
    q, c = req.questions
    assert '"question": "Is it?"' in q.instructions
    assert '"text": "first"' in c.option_descriptions[0] and c.option_descriptions[1] == ""
