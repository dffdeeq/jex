"""Jev-compatible request/response schema.

A request is a ``state`` (string, JSON object or array) plus a map of named,
typed ``questions``. Every question has a closed answer space that is known
before the model runs, so the output can never fall outside it:

* ``noul``   - yes/no, returns P(true)
* ``choice`` - one of 2..255 named options (``criteria`` maps id -> description)
* ``score``  - one of 2..10 ordered levels (``criteria`` is a list of level descriptions)

The response mirrors the public Jev / Clef format so the two are swappable.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

QUESTION_TYPES = ("noul", "choice", "score")
MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS, MAX_SCORE_LEVELS = 2, 10


class SchemaError(ValueError):
    """Raised for a request that violates the typed schema (HTTP 422)."""


@dataclass(frozen=True)
class Question:
    name: str
    type: str
    instructions: str
    # Option ids in the order the caller declared them.
    option_ids: tuple[str, ...]
    # Human readable description for every option (may be empty strings).
    option_descriptions: tuple[str, ...]

    @property
    def num_options(self) -> int:
        return len(self.option_ids)


@dataclass(frozen=True)
class Request:
    state: Any
    questions: tuple[Question, ...]
    model: str = "jex-latest"

    def state_text(self) -> str:
        return render_state(self.state)


@dataclass
class Answer:
    question: Question
    probabilities: list[float]
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        q, p = self.question, self.probabilities
        top = max(range(len(p)), key=p.__getitem__)
        if q.type == "noul":
            return {"type": "noul", "noul": round(p[0], 6)}
        probs = {oid: round(pi, 6) for oid, pi in zip(q.option_ids, p)}
        out: dict[str, Any] = {"type": q.type}
        if q.type == "choice":
            out["choice"] = q.option_ids[top]
        else:
            out["score"] = round(sum(i * pi for i, pi in enumerate(p)), 6)
            out["legend"] = dict(zip(q.option_ids, q.option_descriptions))
        out["probabilities"] = probs
        out["confidence"] = round(p[top], 6)
        return out


def render_state(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=1, sort_keys=False)


def parse_question(name: str, spec: dict[str, Any]) -> Question:
    if not isinstance(spec, dict):
        raise SchemaError(f"question {name!r}: spec must be an object")
    qtype = spec.get("type")
    if qtype not in QUESTION_TYPES:
        raise SchemaError(f"question {name!r}: type must be one of {QUESTION_TYPES}, got {qtype!r}")
    instructions = spec.get("instructions") or ""
    if not isinstance(instructions, str):
        raise SchemaError(f"question {name!r}: instructions must be a string")
    criteria = spec.get("criteria")

    if qtype == "noul":
        # Optional {"true": "...", "false": "..."} descriptions, as in Clef.
        desc = criteria if isinstance(criteria, dict) else {}
        ids = ("true", "false")
        descs = (str(desc.get("true", "")), str(desc.get("false", "")))
        if not instructions:
            raise SchemaError(f"question {name!r}: noul requires instructions")
    elif qtype == "choice":
        if isinstance(criteria, dict):
            ids = tuple(str(k) for k in criteria)
            descs = tuple(str(v or "") for v in criteria.values())
        elif isinstance(criteria, list):
            ids = tuple(str(k) for k in criteria)
            descs = tuple("" for _ in criteria)
        else:
            raise SchemaError(f"question {name!r}: choice criteria must be an object or list")
        if not 2 <= len(ids) <= MAX_CHOICE_OPTIONS:
            raise SchemaError(f"question {name!r}: choice needs 2..{MAX_CHOICE_OPTIONS} options")
        if len(set(ids)) != len(ids):
            raise SchemaError(f"question {name!r}: duplicate option ids")
    else:
        if not isinstance(criteria, list):
            raise SchemaError(f"question {name!r}: score criteria must be a list of levels")
        if not MIN_SCORE_LEVELS <= len(criteria) <= MAX_SCORE_LEVELS:
            raise SchemaError(
                f"question {name!r}: score needs {MIN_SCORE_LEVELS}..{MAX_SCORE_LEVELS} levels"
            )
        ids = tuple(str(i) for i in range(len(criteria)))
        descs = tuple(str(c) for c in criteria)
    return Question(name, qtype, instructions, ids, descs)


def parse_request(payload: dict[str, Any]) -> Request:
    if not isinstance(payload, dict):
        raise SchemaError("request must be a JSON object")
    if "state" not in payload:
        raise SchemaError("request needs a 'state'")
    questions = payload.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise SchemaError("request needs a non-empty 'questions' object")
    parsed = tuple(parse_question(str(n), s) for n, s in questions.items())
    return Request(state=payload["state"], questions=parsed, model=payload.get("model", "jex-latest"))


def build_response(model_name: str, answers: list[Answer], input_tokens: int) -> dict[str, Any]:
    for a in answers:
        if len(a.probabilities) != a.question.num_options or not math.isclose(
            sum(a.probabilities), 1.0, abs_tol=1e-3
        ):
            raise AssertionError(f"invalid distribution for {a.question.name}")
    return {
        "model": model_name,
        "answers": {a.question.name: a.to_json() for a in answers},
        # Nothing is generated: the answer is read off the forward pass.
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
    }
