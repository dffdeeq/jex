"""HTTP API, wire-compatible with Jev / Clef.

    POST /v1/systemone          {"model", "state", "questions"} -> {"model", "answers", "usage"}
    POST /v1/systemone/stream   same request, NDJSON lines {"name", "answer"} in completion order
    POST /v1/ask                {"state", "name", "question"} -> one answer (async, cached state)
    GET  /v1/models

    python -m jex.server --checkpoint artifacts/jex-head --port 8000
"""

from __future__ import annotations

import argparse
import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from .engine import AsyncEngine
from .model import JexModel
from .schema import SchemaError


def create_app(model: JexModel, **engine_kwargs) -> FastAPI:
    engine = AsyncEngine(model, **engine_kwargs)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await engine.start()
        yield
        await engine.stop()

    app = FastAPI(title="jex", lifespan=lifespan)
    app.state.engine = engine

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"data": [{"id": model.name, "backbone": model.backbone.name, "head": model.head is not None}]}

    @app.post("/v1/systemone")
    async def systemone(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return await engine.evaluate(payload)
        except SchemaError as e:
            raise HTTPException(status_code=422, detail=str(e))

    @app.post("/v1/systemone/stream")
    async def systemone_stream(payload: dict[str, Any]):
        try:
            stream = engine.stream(payload)
            first = await anext(stream)
        except SchemaError as e:
            raise HTTPException(status_code=422, detail=str(e))

        async def lines():
            yield json.dumps({"name": first[0], "answer": first[1]}) + "\n"
            async for name, answer in stream:
                yield json.dumps({"name": name, "answer": answer}) + "\n"

        return StreamingResponse(lines(), media_type="application/x-ndjson")

    @app.post("/v1/ask")
    async def ask(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            answer = await engine.ask(payload.get("state"), str(payload.get("name", "q")), payload.get("question"))
        except SchemaError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return {"model": model.name, "name": payload.get("name", "q"), "answer": answer}

    return app


def main():
    import uvicorn

    from .backbone import Backbone

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", help="directory written by scripts/train_head.py")
    ap.add_argument("--backbone", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    model = JexModel.load(args.checkpoint) if args.checkpoint else JexModel(Backbone(args.backbone))
    uvicorn.run(create_app(model), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
