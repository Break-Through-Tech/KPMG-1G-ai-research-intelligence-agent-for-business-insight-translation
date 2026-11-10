"""Thin HTTP layer over the existing pipeline so the web UI can call it.

    pip install fastapi uvicorn
    uvicorn api:app --app-dir src --reload --port 8000

Run from the repo root after parse_and_chunk.py and build_index.py, same as summarize.py.
It reuses Retriever and summarize() as-is; no pipeline logic lives here.
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from retrieve import Retriever
from summarize import DEFAULT_K, make_llm, summarize

DUP_CITE = re.compile(r"(\[arXiv:[^\]]+\])(?:\s*\1)+")

app = FastAPI(title="KPMG 1G Research Assistant API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the embedding model, index and LLM once, not per request.
_retriever: Retriever | None = None
_llm = None


def _pipeline():
    global _retriever, _llm
    if _retriever is None:
        _retriever = Retriever(Path("chroma_db"))
        _llm = make_llm()
    return _retriever, _llm


class ChatRequest(BaseModel):
    question: str
    k: int = DEFAULT_K


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/chat")
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question is empty.")
    try:
        retriever, llm = _pipeline()
        answer = summarize(req.question, retriever.search, llm, req.k)
    except SystemExit as exc:  # the pipeline uses SystemExit for setup problems (no index, no key)
        raise HTTPException(503, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Pipeline error: {exc}")

    cited = set(answer.cited_ids)
    sources = [
        {
            "arxivId": h.arxiv_id,
            "title": h.title,
            "pages": f"{h.page_start}-{h.page_end}",
            "score": round(float(getattr(h, "score", getattr(h, "similarity", 0.0))), 3),
            "passage": " ".join(h.text.split())[:600],
            "cited": h.arxiv_id in cited,
        }
        for h in answer.hits
    ]
    return {
        "answer": DUP_CITE.sub(r"\1", answer.text),
        "sources": sources,
        "unknownCitations": answer.unknown_citations,
        "abstained": getattr(answer, "abstained", False),  # exists on the refine-retrieval branch
    }
