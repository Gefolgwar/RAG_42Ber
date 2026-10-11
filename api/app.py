"""Local FastAPI server exposing search and answer."""

from typing import List, Any

from fastapi import FastAPI
from pydantic import BaseModel

from src.retrieval import chunk_to_source, load_kb

app = FastAPI(title="RAG Against the Machine")


class SearchRequest(BaseModel):
    """Search request body."""

    query: str
    k: int = 5


class SearchResponse(BaseModel):
    """Search response body."""

    query: str
    sources: List[dict[str, Any]] = []


class AnswerResponse(BaseModel):
    """Answer response body."""

    query: str
    answer: str


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    """Return top-k sources for a query."""
    try:
        kb = load_kb("data/processed")
        if kb is None or not req.query.strip() or req.k <= 0:
            return SearchResponse(query=req.query, sources=[])
        pairs = kb.search_bm25(req.query, k=req.k)
        srcs = [chunk_to_source(ch).model_dump() for ch, _ in pairs]
        return SearchResponse(query=req.query, sources=srcs)
    except Exception as exc:
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=500, content={"error": str(exc)})  # type: ignore


@app.post("/answer", response_model=AnswerResponse)
def answer(req: SearchRequest) -> AnswerResponse:
    """Answer a query from top-k context."""
    try:
        from fastapi.responses import JSONResponse

        from src.generator import get_generator

        kb = load_kb("data/processed")
        if kb is None or not req.query.strip():
            return AnswerResponse(query=req.query, answer="")
        pairs = kb.search_bm25(req.query, k=req.k)
        chunks = [ch for ch, _ in pairs]
        text = get_generator().generate(req.query, chunks)
        return AnswerResponse(query=req.query, answer=text)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})  # type: ignore
