"""Local FastAPI server exposing search and answer (Bonus 5)."""

from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from src.cli import _chunk_to_source, _load_kb

app = FastAPI(title="RAG Against the Machine")


class SearchRequest(BaseModel):
    """Search request body."""

    query: str
    k: int = 5


class SearchResponse(BaseModel):
    """Search response body."""

    query: str
    sources: List[dict] = []


class AnswerResponse(BaseModel):
    """Answer response body."""

    query: str
    answer: str


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    """Return top-k sources for a query.

    Args:
        req: Query plus depth.

    Returns:
        Query echo with source list (500 on retrieval failure).
    """
    try:
        kb = _load_kb("data/processed")
        if kb is None or not req.query.strip() or req.k <= 0:
            return SearchResponse(query=req.query, sources=[])
        pairs = kb.search_bm25(req.query, k=req.k)
        srcs = [_chunk_to_source(ch).model_dump() for ch, _ in pairs]
        return SearchResponse(query=req.query, sources=srcs)
    except Exception as exc:
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=500, content={"error": str(exc)})  # type: ignore


@app.post("/answer", response_model=AnswerResponse)
def answer(req: SearchRequest) -> AnswerResponse:
    """Answer a query from top-k retrieved context.

    Args:
        req: Query plus depth.

    Returns:
        Query echo with generated answer (500 on failure).
    """
    try:
        from fastapi.responses import JSONResponse

        from src.generator import get_generator

        kb = _load_kb("data/processed")
        if kb is None or not req.query.strip():
            return AnswerResponse(query=req.query, answer="")
        pairs = kb.search_bm25(req.query, k=req.k)
        chunks = [ch for ch, _ in pairs]
        text = get_generator().generate(req.query, chunks)
        return AnswerResponse(query=req.query, answer=text)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})  # type: ignore
