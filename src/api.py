"""Thin HTTP layer over the existing pipeline so the web UI can call it.

    pip install fastapi uvicorn
    uvicorn api:app --app-dir src --reload --port 8000

Run from the repo root after parse_and_chunk.py and build_index.py, same as summarize.py.
It reuses Retriever and summarize() as-is; no pipeline logic lives here.
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
import threading
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from parse_and_chunk import FILENAME_RE
from retrieve import Retriever
from summarize import DEFAULT_K, make_llm, summarize

ROOT = Path(__file__).resolve().parent.parent  # repo root, wherever uvicorn is started from
DATA_DIR = ROOT / "data"  # same folder parse_and_chunk.py reads; data/*.pdf is gitignored
MAX_UPLOAD_MB = 40
_index_lock = threading.Lock()

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
        _retriever = Retriever(ROOT / "chroma_db")
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


def _paper_info(pdf: Path) -> dict:
    m = FILENAME_RE.match(pdf.stem)
    return {
        "file": pdf.name,
        "arxivId": m["arxiv_id"] if m else None,
        "title": m["title"] if m else pdf.stem,
    }


@app.get("/papers")
def papers():
    """The PDFs the index is built from."""
    return {"papers": [_paper_info(p) for p in sorted(DATA_DIR.glob("*.pdf"))]}


def _safe_name(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^\w.\- ]+", "", stem).strip() or "upload"
    return f"{stem[:150]}.pdf"


def _run(script: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, script], cwd=ROOT, capture_output=True, text=True)


def _fail(script: str, result: subprocess.CompletedProcess) -> RuntimeError:
    tail = (result.stderr or result.stdout).strip().splitlines()[-3:]
    return RuntimeError(f"{script} failed: {' | '.join(tail)}")


def _parse_report() -> dict[str, dict]:
    report = ROOT / "data/processed/parse_report.csv"
    if not report.exists():
        return {}
    with open(report, encoding="utf-8") as fh:
        return {r["source_file"]: r for r in csv.DictReader(fh)}


def _rebuild_index(new_files: list[Path]) -> tuple[dict[str, dict], list[Path]]:
    """Re-run the team's own scripts, exactly as the README does.

    parse_and_chunk.py exits 1 when any PDF yields 0 chunks (scanned, protected, or a
    near-duplicate of a paper already indexed). Uploads like that are removed and parsing
    re-runs, so data/ never holds a PDF that breaks `parse_and_chunk && build_index`.
    """
    parse = "src/parse_and_chunk.py"
    result = _run(parse)
    rows = _parse_report()
    rejected = [p for p in new_files if int(rows.get(p.name, {}).get("chunks") or 0) == 0]
    if rejected:
        for p in rejected:
            p.unlink(missing_ok=True)
        result = _run(parse)
    if result.returncode != 0:
        raise _fail(parse, result)
    build = _run("src/build_index.py")
    if build.returncode != 0:
        raise _fail("src/build_index.py", build)
    return rows, rejected


@app.post("/upload")
def upload(files: list[UploadFile] = File(...)):
    """Add PDFs to data/ and rebuild the index so they are searchable.
    Name arXiv papers like 2608.20319v1_Title.pdf so citations link to arXiv."""
    global _retriever
    saved: list[Path] = []
    try:
        for f in files:
            if not (f.filename or "").lower().endswith(".pdf"):
                raise HTTPException(400, f"{f.filename} isn't a PDF.")
            data = f.file.read()
            if not data.startswith(b"%PDF"):
                raise HTTPException(400, f"{f.filename} doesn't look like a valid PDF.")
            if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
                raise HTTPException(400, f"{f.filename} is over {MAX_UPLOAD_MB} MB.")
            dest = DATA_DIR / _safe_name(f.filename)
            if dest.exists():
                raise HTTPException(409, f"{dest.name} is already in the index.")
            dest.write_bytes(data)
            saved.append(dest)

        if not _index_lock.acquire(blocking=False):
            raise HTTPException(409, "The index is already being rebuilt. Try again in a minute.")
        try:
            rows, rejected = _rebuild_index(saved)
            _retriever = None  # reload so the next question sees the new chunks
        finally:
            _index_lock.release()
    except Exception as exc:
        for p in saved:  # don't leave half-added files behind
            p.unlink(missing_ok=True)
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(500, f"Couldn't index the upload: {exc}")

    added, skipped = [], []
    for p in saved:
        r = rows.get(p.name, {})
        info = {**_paper_info(p), "chunks": int(r.get("chunks") or 0), "status": r.get("status", "unknown")}
        (skipped if p in rejected else added).append(info)
    return {"added": added, "skipped": skipped}
