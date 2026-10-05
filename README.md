# jex: System One модель в стиле Jev поверх обычной LLM

`jex` превращает замороженную LLM в модель принятия решений, как
[Jev от TypeSafe](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
или [Clef от Cloudflare](https://blog.cloudflare.com/clef-decision-models/).
Модель получает `state` и набор типизированных вопросов (`noul` / `choice` / `score`)
и за **один prefill-проход без генерации** возвращает на каждый вопрос
калиброванное распределение по заранее заданным вариантам.

Ответы асинхронны в буквальном смысле:

* **внутри запроса** вопросы упакованы в изолированные ветки (древовидная маска
  внимания и позиции, которые в каждой ветке начинаются заново сразу после state).
  Каждый ответ считается так же, как при отдельном вызове: порядок вопросов и
  добавление новых ничего не меняют;
* **между запросами** state кодируется один раз и хранится в KV-кэше. Вопрос,
  пришедший позже, оплачивает только свои токены. Движок непрерывно
  микробатчит всё, что накопилось в очереди, и закрывает future каждого
  вопроса отдельно (есть стриминг в порядке готовности);
* **живые сессии**: в state можно дописывать события (чат, логи, игра),
  пересчитываются только новые токены. Вопрос видит ровно те события, которые
  были добавлены до него.

Подробный ресерч (Jev, Laya, Clef, предполагаемая архитектура Jev, варианты
сборки), результаты экспериментов и план масштабирования лежат в
[`docs/RESEARCH.md`](docs/RESEARCH.md).

## Быстрый старт

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[server,dev]"
pytest                                    # 37 тестов, крошечная случайная модель, ~15 с
```

```python
from jex.backbone import Backbone
from jex.model import JexModel

model = JexModel(Backbone("Qwen/Qwen2.5-0.5B-Instruct"))   # zero-shot, без обучения
model.predict({
    "state": {"subject": "Duplicate charge on invoice #4411",
              "body": "We were billed twice for March. Refund or we cancel."},
    "questions": {
        "department": {"type": "choice", "instructions": "Which department should handle this?",
                       "criteria": {"billing": "invoices, refunds", "technical": "bugs", "sales": "upgrades"}},
        "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel?"},
        "urgency": {"type": "score", "instructions": "How urgent is this?",
                    "criteria": ["not urgent", "somewhat urgent", "urgent", "critical"]},
    },
})
# {'answers': {'department': {'type': 'choice', 'choice': 'billing', 'probabilities': {...}, 'confidence': 0.98},
#              'churn_risk': {'type': 'noul', 'noul': 0.76}, 'urgency': {'type': 'score', 'score': 1.56, ...}},
#  'usage': {'input_tokens': 210, 'output_tokens': 0}}
```

Асинхронный движок:

```python
from jex.engine import AsyncEngine

async with AsyncEngine(model) as engine:
    answer = await engine.ask(state, "churn_risk", {"type": "noul", "instructions": "..."})
    async for name, answer in engine.stream(payload):      # в порядке готовности
        ...
    await engine.open_session("game", "Game log. Turn 1: ...")
    engine.append("game", " Turn 2: a dragon attacks ...")
    await engine.ask_session("game", "danger", {"type": "noul", "instructions": "Is the player in danger?"})
```

HTTP-API, совместимое с Jev и Clef:

```bash
python -m jex.server --checkpoint artifacts/jex-head      # или --backbone <hf id> для zero-shot
curl localhost:8000/v1/systemone -d @request.json
```

## Обучение головы (дистилляция из модели побольше)

```bash
python scripts/extract.py      # данные + признаки студента (0.5B) + soft-labels учителя (1.5B)
python scripts/train_head.py   # CE + Brier + KD, в конце RLCD → artifacts/jex-head
python scripts/evaluate.py     # точность/ECE на знакомых и отложенных задачах
python scripts/bench_latency.py
python scripts/bench_async.py
```

## Устройство

| файл | что делает |
|---|---|
| `jex/schema.py` | запрос/ответ в формате Jev, валидация типов (choice ≤ 255 вариантов, score 2–10 уровней) |
| `jex/render.py` | общий префикс со state и ветка на каждый вопрос; однотокенные метки вариантов |
| `jex/packing.py` | древовидная маска внимания и перезапуск позиций в ветках, упаковка поверх кэша |
| `jex/backbone.py` | замороженная causal LM: prefill, ветки по кэшу, дописывание state, verbalizer-логиты |
| `jex/head.py` | обучаемая голова в духе Clef/Laya: evidence routing, конкуренция вариантов, гейт к лексическому prior |
| `jex/rlcd.py` | RLCD: гауссова политика на логитах, строго собственные scoring rules, групповой baseline |
| `jex/training.py` | обучение головы: gold + дистилляция калиброванного учителя + Brier + RLCD |
| `jex/engine.py` | async-движок: LRU KV-кэш состояний, непрерывный микробатчинг, стриминг, сессии |
| `jex/server.py` | FastAPI: `/v1/systemone`, `/v1/systemone/stream`, `/v1/ask` |
