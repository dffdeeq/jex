"""Typed-decision datasets built from public text-classification corpora.

Every record is a Jev request (state + several typed questions) plus gold
labels for the questions that have them. Questions without gold labels are
answered by the teacher only (distillation targets).

Train tasks and held-out tasks are disjoint, so the evaluation measures what
matters for a System One model: answering *new* schemas defined at request
time.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

from datasets import load_dataset


@dataclass
class Record:
    task: str
    state: Any
    questions: dict[str, dict[str, Any]]
    labels: dict[str, int] = field(default_factory=dict)  # gold option index per question

    def payload(self) -> dict[str, Any]:
        return {"state": self.state, "questions": self.questions}


def _clip(text: str, n: int = 700) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + " ..."


def _choice(rng, instructions: list[str], options: list[tuple[str, str]], gold: int | None, describe_p=0.7):
    """Choice question with shuffled options and optional descriptions; returns (spec, gold index)."""
    order = list(range(len(options)))
    rng.shuffle(order)
    with_desc = rng.random() < describe_p
    criteria = {options[i][0]: (options[i][1] if with_desc else "") for i in order}
    spec = {"type": "choice", "instructions": rng.choice(instructions), "criteria": criteria}
    return spec, (order.index(gold) if gold is not None else None)


def _noul(rng, instructions: list[str]):
    return {"type": "noul", "instructions": rng.choice(instructions)}


def _score(rng, instructions: list[str], levels: list[str]):
    return {"type": "score", "instructions": rng.choice(instructions), "criteria": list(levels)}


# --------------------------------------------------------------------- tasks

AG_TOPICS = [
    ("world", "international news, politics, conflicts, diplomacy"),
    ("sports", "games, athletes, teams, tournaments"),
    ("business", "companies, markets, economy, earnings"),
    ("sci_tech", "science, technology, computing, the internet"),
]
EMOTIONS = [
    ("sadness", "feeling sad, down, hurt"),
    ("joy", "feeling happy, glad, content"),
    ("love", "feeling affection, caring, romance"),
    ("anger", "feeling angry, annoyed, irritated"),
    ("fear", "feeling afraid, anxious, nervous"),
    ("surprise", "feeling surprised, amazed, shocked"),
]
DBPEDIA = [
    ("company", ""), ("school", "educational institution"), ("artist", ""), ("athlete", ""),
    ("politician", "office holder"), ("transport", "mean of transportation"), ("building", ""),
    ("nature", "natural place"), ("village", ""), ("animal", ""), ("plant", ""), ("album", ""),
    ("film", ""), ("book", "written work"),
]
SENTIMENT5 = ["very negative", "negative", "neutral", "positive", "very positive"]
# Same order as SetFit/TREC-QC label_coarse.
TREC = [
    ("description", "asks for a definition, description, manner or reason"),
    ("entity", "asks for a thing: animal, color, food, product, term, ..."),
    ("abbreviation", "asks what an abbreviation stands for"),
    ("human", "asks about a person or group of people"),
    ("numeric", "asks for a number, date, distance, money, count, ..."),
    ("location", "asks for a place: city, country, mountain, ..."),
]
STARS = ["1 star: terrible", "2 stars: poor", "3 stars: average", "4 stars: good", "5 stars: excellent"]

# Auxiliary questions without gold labels: the teacher provides soft targets.
AUX = [
    lambda r: _noul(r, ["Does the text mention a specific named person?", "Is a particular person named in the text?"]),
    lambda r: _noul(r, ["Is the text about money, prices or finance?", "Does the text deal with money or finance?"]),
    lambda r: _noul(r, ["Is the tone of the text negative?", "Does the author sound unhappy or critical?"]),
    lambda r: _choice(r, ["What is the overall tone?", "How would you describe the tone of the text?"],
                      [("positive", "approving, happy"), ("neutral", "factual, balanced"), ("negative", "critical, unhappy")], None)[0],
    lambda r: _choice(r, ["What is the main subject area?", "Which area is the text mostly about?"],
                      [("politics", ""), ("technology", ""), ("entertainment", "film, music, tv"), ("sports", ""),
                       ("business", ""), ("everyday_life", "personal matters")], None)[0],
    lambda r: _score(r, ["How formal is the language?", "Rate the formality of the writing."],
                     ["very informal", "neutral", "very formal"]),
]


def _sst2(row, rng):
    if rng.random() < 0.5:
        q = _noul(rng, ["Is the review positive?", "Does the reviewer like the movie?", "Is the sentiment positive?"])
        return {"text": _clip(row["text"])}, {"sentiment": q}, {"sentiment": 0 if row["label"] == 1 else 1}
    spec, gold = _choice(rng, ["What is the sentiment of the review?", "Is the review positive or negative?"],
                         [("negative", "the reviewer dislikes it"), ("positive", "the reviewer likes it")], row["label"])
    return {"review": _clip(row["text"])}, {"sentiment": spec}, {"sentiment": gold}


def _ag(row, rng):
    spec, gold = _choice(rng, ["What is the topic of the article?", "Which section does this news belong to?"],
                         AG_TOPICS, row["label"])
    qs, labels = {"topic": spec}, {"topic": gold}
    if rng.random() < 0.5:
        qs["is_sports"] = _noul(rng, ["Is this article about sports?", "Is the news story about sports?"])
        labels["is_sports"] = 0 if row["label"] == 1 else 1
    return {"article": _clip(row["text"])}, qs, labels


def _emotion(row, rng):
    spec, gold = _choice(rng, ["Which emotion does the writer express?", "What is the dominant emotion?"],
                         EMOTIONS, row["label"])
    return _clip(row["text"]), {"emotion": spec}, {"emotion": gold}


def _sst5(row, rng):
    spec = _score(rng, ["How positive is the review?", "Rate the sentiment of the review."], SENTIMENT5)
    return {"review": _clip(row["text"])}, {"rating": spec}, {"rating": row["label"]}


def _dbpedia(row, rng):
    spec, gold = _choice(rng, ["What kind of entity is described?", "Which category does the entry belong to?"],
                         DBPEDIA, row["label"])
    return {"title": row["title"], "content": _clip(row["content"], 500)}, {"category": spec}, {"category": gold}


def _subj(row, rng):
    q = _noul(rng, ["Is the sentence a subjective opinion rather than an objective description?",
                    "Does the sentence express the author's opinion?"])
    # SetFit/subj: label 1 = subjective
    return _clip(row["text"]), {"subjective": q}, {"subjective": 0 if row["label"] == 1 else 1}


# Held-out tasks: never used to train the head.
def _rotten(row, rng):
    q = {"type": "noul", "instructions": "Is this movie review positive?"}
    return {"review": _clip(row["text"])}, {"positive": q}, {"positive": 0 if row["label"] == 1 else 1}


def _spam(row, rng):
    q = {"type": "noul", "instructions": "Is this email spam (unsolicited bulk or advertising mail)?"}
    state = {"subject": _clip(row["subject"] or "", 150), "message": _clip(row["message"] or "", 600)}
    return state, {"spam": q}, {"spam": 0 if row["label"] == 1 else 1}


def _trec(row, rng):
    spec = {"type": "choice", "instructions": "What type of answer does the question ask for?",
            "criteria": dict(TREC)}
    return {"question": row["text"]}, {"answer_type": spec}, {"answer_type": row["label_coarse"]}


def _yelp(row, rng):
    spec = {"type": "score", "instructions": "How many stars did the customer give?", "criteria": STARS}
    return {"review": _clip(row["text"], 600)}, {"stars": spec}, {"stars": row["label"]}


TRAIN_TASKS: dict[str, tuple[str, Callable]] = {
    "sst2": ("SetFit/sst2", _sst2),
    "ag_news": ("SetFit/ag_news", _ag),
    "emotion": ("SetFit/emotion", _emotion),
    "sst5": ("SetFit/sst5", _sst5),
    "dbpedia": ("fancyzhx/dbpedia_14", _dbpedia),
    "subj": ("SetFit/subj", _subj),
}
HELDOUT_TASKS: dict[str, tuple[str, Callable]] = {
    "rotten_tomatoes": ("cornell-movie-review-data/rotten_tomatoes", _rotten),
    "enron_spam": ("SetFit/enron_spam", _spam),
    "trec": ("SetFit/TREC-QC", _trec),
    "yelp_stars": ("Yelp/yelp_review_full", _yelp),
}


def build(task: str, split: str, n: int, seed: int = 0, aux_max: int = 0, offset: int = 0) -> list[Record]:
    """``n`` records from ``split``; ``offset`` skips records of the same shuffled order,
    so e.g. a dev set can be carved from train without overlap."""
    source, fn = {**TRAIN_TASKS, **HELDOUT_TASKS}[task]
    ds = load_dataset(source, split=split).shuffle(seed=seed).select(range(offset, offset + n))
    rng = random.Random(f"{task}-{split}-{seed}-{offset}")
    records = []
    for row in ds:
        state, qs, labels = fn(row, rng)
        for k in range(rng.randint(0, aux_max)):
            qs[f"aux_{k}"] = rng.choice(AUX)(rng)
        names = list(qs)
        rng.shuffle(names)  # question order must not matter
        records.append(Record(task, state, {k: qs[k] for k in names}, labels))
    return records
