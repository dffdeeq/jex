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
from functools import lru_cache
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


def _noul_pm(rng, positive: list[str], negative: list[str], is_positive: bool, flip_p: float = 0.4):
    """Yes/no question asked in either polarity, so "yes" is not tied to one class
    (counters the yes/no base-rate bias a small head picks up). Returns (spec, gold)."""
    if rng.random() < flip_p:
        return _noul(rng, negative), (1 if is_positive else 0)
    return _noul(rng, positive), (0 if is_positive else 1)


def _choice_subset(rng, instructions: list[str], names: list[str], gold: int, k_min: int, k_max: int):
    """Choice over a random subset of a large label set that always contains the gold label."""
    k = min(rng.randint(k_min, k_max), len(names))
    others = rng.sample([i for i in range(len(names)) if i != gold], k - 1)
    idx = others + [gold]
    rng.shuffle(idx)
    spec = {"type": "choice", "instructions": rng.choice(instructions), "criteria": {names[i]: "" for i in idx}}
    return spec, idx.index(gold)


TEXT_KEYS = ["text", "content", "message", "body", "review", "comment", "post", "input", "document"]


def _rewrap(state: Any, rng) -> Any:
    """Randomize the surface form of a single-text state (plain string, different
    keys, extra metadata) so the head cannot key on it."""
    if isinstance(state, dict) and len(state) == 1:
        text = next(iter(state.values()))
    elif isinstance(state, str):
        text = state
    else:
        return state
    r = rng.random()
    if r < 0.3:
        return text
    out = {rng.choice(TEXT_KEYS): text}
    if r > 0.8:
        out = {"id": f"{rng.randrange(10**6):06d}", "source": rng.choice(["web", "app", "email", "forum", "api"]), **out}
    return out


@lru_cache(maxsize=None)
def _label_names(source: str) -> list[str]:
    ds = load_dataset(source, split="train")
    names = dict(zip(ds["label"], ds["label_text"]))
    return [names[i] for i in range(len(names))]


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
# Level descriptions must not contain numbers: levels are labeled 0-9 for the model,
# and "1 star" next to label "0" confuses it (v1 of this task did exactly that).
STARS = ["terrible", "poor", "average", "good", "excellent"]

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
        q, gold = _noul_pm(rng, ["Is the review positive?", "Does the reviewer like the movie?", "Is the sentiment positive?"],
                           ["Is the review negative?", "Does the reviewer dislike the movie?"], row["label"] == 1)
        return {"text": _clip(row["text"])}, {"sentiment": q}, {"sentiment": gold}
    spec, gold = _choice(rng, ["What is the sentiment of the review?", "Is the review positive or negative?"],
                         [("negative", "the reviewer dislikes it"), ("positive", "the reviewer likes it")], row["label"])
    return {"review": _clip(row["text"])}, {"sentiment": spec}, {"sentiment": gold}


def _ag(row, rng):
    spec, gold = _choice(rng, ["What is the topic of the article?", "Which section does this news belong to?"],
                         AG_TOPICS, row["label"])
    qs, labels = {"topic": spec}, {"topic": gold}
    if rng.random() < 0.5:
        qs["is_sports"], labels["is_sports"] = _noul_pm(
            rng, ["Is this article about sports?", "Is the news story about sports?"],
            ["Is this article about something other than sports?"], row["label"] == 1)
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
    # SetFit/subj: label 1 = subjective
    q, gold = _noul_pm(rng, ["Is the sentence a subjective opinion rather than an objective description?",
                             "Does the sentence express the author's opinion?"],
                       ["Is the sentence an objective, factual description?"], row["label"] == 1)
    return _clip(row["text"]), {"subjective": q}, {"subjective": gold}


def _toxic(row, rng):
    q, gold = _noul_pm(rng, ["Is the comment toxic or abusive?", "Does the comment contain insults or hostility?"],
                       ["Is the comment civil and non-toxic?"], row["label"] == 1)
    return _clip(row["text"]), {"toxic": q}, {"toxic": gold}


def _bbc(row, rng):
    spec, gold = _choice(rng, ["Which section of the news site is this?", "What is the article about?"],
                         [("tech", "technology"), ("business", ""), ("sport", ""), ("entertainment", ""),
                          ("politics", "")], row["label"])
    return {"article": _clip(row["text"])}, {"section": spec}, {"section": gold}


def _hate(row, rng):
    spec, gold = _choice(rng, ["How would a moderator classify this tweet?", "Which category fits the tweet?"],
                         [("hate_speech", "attacks a group based on identity"),
                          ("offensive", "rude or vulgar but not targeted hate"), ("neither", "acceptable")],
                         row["label"])
    return _clip(row["text"]), {"moderation": spec}, {"moderation": gold}


def _tweet_sentiment(row, rng):
    spec, gold = _choice(rng, ["What is the sentiment of the tweet?", "How does the author feel?"],
                         [("negative", ""), ("neutral", ""), ("positive", "")], row["label"])
    return _clip(row["text"]), {"sentiment": spec}, {"sentiment": gold}


def _massive(row, rng):
    names = _label_names("SetFit/amazon_massive_intent_en-US")
    # 6..40 options: beyond 26 there is no single-token label, which trains the head-only path.
    spec, gold = _choice_subset(rng, ["What does the user want the assistant to do?", "Which intent is this?"],
                                names, row["label"], 6, 40)
    return {"utterance": row["text"]}, {"intent": spec}, {"intent": gold}


def _newsgroups(row, rng):
    names = _label_names("SetFit/20_newsgroups")
    spec, gold = _choice_subset(rng, ["Which newsgroup was this posted to?", "Where does this post belong?"],
                                names, row["label"], 4, 12)
    return _clip(row["text"], 600), {"group": spec}, {"group": gold}


def _insincere(row, rng):
    q, gold = _noul_pm(rng, ["Is this question insincere, i.e. a statement meant to provoke rather than a real question?"],
                       ["Is this a sincere question that genuinely seeks an answer?"], row["label"] == 1)
    return {"question": row["text"]}, {"insincere": q}, {"insincere": gold}


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
    spec = {"type": "score", "instructions": "How satisfied is the customer with the business?", "criteria": STARS}
    return {"review": _clip(row["text"], 600)}, {"stars": spec}, {"stars": row["label"]}


TRAIN_TASKS: dict[str, tuple[str, Callable]] = {
    "sst2": ("SetFit/sst2", _sst2),
    "ag_news": ("SetFit/ag_news", _ag),
    "emotion": ("SetFit/emotion", _emotion),
    "sst5": ("SetFit/sst5", _sst5),
    "dbpedia": ("fancyzhx/dbpedia_14", _dbpedia),
    "subj": ("SetFit/subj", _subj),
    "toxic": ("SetFit/toxic_conversations", _toxic),
    "bbc_news": ("SetFit/bbc-news", _bbc),
    "hate_speech": ("SetFit/hate_speech_offensive", _hate),
    "tweet_sentiment": ("SetFit/tweet_sentiment_extraction", _tweet_sentiment),
    "massive_intent": ("SetFit/amazon_massive_intent_en-US", _massive),
    "newsgroups": ("SetFit/20_newsgroups", _newsgroups),
    "insincere": ("SetFit/insincere-questions", _insincere),
}
# Rare-positive binary tasks are sampled class-balanced.
BALANCED = {"toxic", "insincere"}
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
    ds = load_dataset(source, split=split).shuffle(seed=seed)
    if task in BALANCED:
        head = ds.select(range(min(len(ds), 200_000)))
        pos = [i for i, y in enumerate(head["label"]) if y == 1]
        neg = [i for i, y in enumerate(head["label"]) if y != 1]
        order = [x for pair in zip(pos, neg) for x in pair]
        ds = head.select(order)
    ds = ds.select(range(offset, offset + n))
    rng = random.Random(f"{task}-{split}-{seed}-{offset}")
    records = []
    for row in ds:
        state, qs, labels = fn(row, rng)
        if task in TRAIN_TASKS:
            state = _rewrap(state, rng)
        for k in range(rng.randint(0, aux_max)):
            qs[f"aux_{k}"] = rng.choice(AUX)(rng)
        names = list(qs)
        rng.shuffle(names)  # question order must not matter
        records.append(Record(task, state, {k: qs[k] for k in names}, labels))
    return records
