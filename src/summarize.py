"""Answer a question from the retrieved chunks, citing the arXiv id behind each claim.

    python src/summarize.py "How close are AI agents to designing their own training algorithms?"
    python src/summarize.py "your question" -k 10 --prompt-version codex_v1 --show-sources

Retrieves the top-k chunks with retrieve.py, sends them to a free-tier Gemini model
through LangChain, and prints an answer where every claim ends in [arXiv:ID].
The model is told to answer only from the retrieved chunks, and any citation to a
paper that was not retrieved is flagged.

Needs GOOGLE_API_KEY in .env (see .env.example). Free-tier limits vary by project,
so check yours in Google AI Studio before running a big batch.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from evaluate_retrieval import paper_ranking
from evidence import evidence_messages, passage_text, render_evidence
from pull_arxiv import base_id

DEFAULT_MODEL = "gemini-3.5-flash-lite"  # existing default; override with --model or GEMINI_MODEL
DEFAULT_K = 5
MAX_CHARS_PER_CHUNK = 1500  # safety cap; normal 120-word chunks are well under it

ARXIV_ID_RE = re.compile(r"(?<![\w.])[0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?(?![\w.])")
# Any [...] holding an arXiv id: [arXiv:A], [A], [arXiv:A, arXiv:B], [arXiv: A; B]
CITATION_BRACKET_RE = re.compile(r"\[[^\[\]]*\d{4}\.\d{4,5}[^\[\]]*\]")

SYSTEM_PROMPT = """You answer questions about AI research for a business audience.

Rules:
- Use ONLY the numbered sources in the user message. Do not use outside knowledge.
- End every sentence that states a finding with its citation in the form [arXiv:ID], using the ID shown for that source.
- If two sources support a claim, cite both: [arXiv:ID1] [arXiv:ID2].
- Keep the paper's own numbers and names exactly as written. Do not round or rename.
- If the sources do not answer the question, say "The retrieved papers do not answer this." and stop.
- Plain language, 3 to 6 sentences."""


@dataclass
class Answer:
    question: str
    text: str
    hits: list = field(default_factory=list)
    retrieved_ids: list[str] = field(default_factory=list)
    cited_ids: list[str] = field(default_factory=list)
    uncited_sources: list[str] = field(default_factory=list)  # retrieved but never cited
    unknown_citations: list[str] = field(default_factory=list)  # cited but never retrieved
    raw_response: str = ""  # codex_v1's unmodified structured output for audit
    evidence: list[dict] = field(default_factory=list)  # accepted claims and exact excerpts
    validation_errors: list[str] = field(default_factory=list)


def format_sources(hits: Sequence, max_chars: int = MAX_CHARS_PER_CHUNK) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        text = passage_text(hit, max_chars)
        blocks.append(
            f"[{i}] ID: {hit.arxiv_id} | {hit.title} | pages {hit.page_start}-{hit.page_end}\n{text}"
        )
    return "\n\n".join(blocks)


def build_messages(question: str, hits: Sequence, *, prompt_version: str = "baseline") -> list[tuple[str, str]]:
    """(role, content) pairs. LangChain chat models accept these directly."""
    if prompt_version == "codex_v1":
        return evidence_messages(question, hits, MAX_CHARS_PER_CHUNK)
    if prompt_version != "baseline":
        raise ValueError(f"Unknown summarization prompt version: {prompt_version}")
    human = f"Sources:\n\n{format_sources(hits)}\n\nQuestion: {question}"
    return [("system", SYSTEM_PROMPT), ("human", human)]


def extract_citations(text: str) -> list[str]:
    """arXiv ids cited inside [...] brackets, in order of first appearance.
    An id mentioned outside a bracket is not counted as a citation."""
    ids: list[str] = []
    for bracket in CITATION_BRACKET_RE.findall(text):
        for cid in ARXIV_ID_RE.findall(bracket):
            if cid not in ids:
                ids.append(cid)
    return ids


def response_text(response) -> str:
    """Support LangChain text properties/methods and text content blocks."""
    text = getattr(response, "text", None)
    # Current LangChain exposes a callable str subclass; read it as a property
    # before considering older method-style messages, avoiding a deprecation.
    if isinstance(text, str):
        return text.strip()
    if callable(text):
        text = text()
    if isinstance(text, str):
        return text.strip()
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(block["text"] for block in content
                         if isinstance(block, dict) and block.get("type") == "text"
                         and isinstance(block.get("text"), str)).strip()
    return ""


def summarize(question: str, search: Callable[[str, int], Sequence], llm, k: int = DEFAULT_K,
              *, prompt_version: str = "baseline") -> Answer:
    """search(question, k) returns hits like retrieve.Retriever.search does.
    llm is any LangChain chat model."""
    if k <= 0:
        raise ValueError("k must be positive")
    if prompt_version not in ("baseline", "codex_v1"):
        raise ValueError(f"Unknown summarization prompt version: {prompt_version}")
    hits = list(search(question, k))
    retrieved = paper_ranking([h.arxiv_id for h in hits])
    if not hits:
        return Answer(question=question, text="No chunks were retrieved, so there is nothing to summarize.")

    raw = response_text(llm.invoke(build_messages(question, hits, prompt_version=prompt_version)))
    evidence, errors = [], []
    if prompt_version == "codex_v1":
        text, evidence, errors = render_evidence(raw, hits, MAX_CHARS_PER_CHUNK)
    else:
        text = raw
    cited = extract_citations(text)
    retrieved_base = {base_id(r) for r in retrieved}
    cited_base = {base_id(c) for c in cited}
    return Answer(
        question=question,
        text=text,
        hits=hits,
        retrieved_ids=retrieved,
        cited_ids=cited,
        uncited_sources=[r for r in retrieved if base_id(r) not in cited_base],
        unknown_citations=[c for c in cited if base_id(c) not in retrieved_base],
        raw_response=raw if prompt_version == "codex_v1" else "",
        evidence=evidence,
        validation_errors=errors,
    )


def make_llm(model: str | None = None):
    # No temperature on purpose: Google's Gemini 3 guide says to leave it at the default.
    from dotenv import load_dotenv
    from langchain_google_genai import ChatGoogleGenerativeAI

    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key.")
    model = model or os.getenv("GEMINI_MODEL") or DEFAULT_MODEL
    return ChatGoogleGenerativeAI(model=model, max_retries=2)


def print_answer(answer: Answer, show_sources: bool = False) -> None:
    print(answer.text)
    print()
    titles = {h.arxiv_id: h.title for h in answer.hits}
    print("Sources:")
    for pid in answer.retrieved_ids:
        mark = "not cited" if pid in answer.uncited_sources else "cited"
        print(f"  {pid}  ({mark})  {titles.get(pid, '')}")
    if answer.unknown_citations:
        print(f"\nWARNING: cites papers that were not retrieved: {', '.join(answer.unknown_citations)}")
    if answer.validation_errors:
        print("\nEvidence validation: " + "; ".join(answer.validation_errors))
    if show_sources:
        print()
        for hit in answer.hits:
            print(f"{hit.rank}. [{hit.score:.3f}] {hit.arxiv_id} p.{hit.page_start}-{hit.page_end}")
            print(f"   {hit.text[:300]}...")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=DEFAULT_K)
    parser.add_argument("--db", type=Path, default=Path("chroma_db"))
    parser.add_argument("--model", default=None, help=f"default: GEMINI_MODEL from .env, else {DEFAULT_MODEL}")
    parser.add_argument("--show-sources", action="store_true", help="also print the retrieved chunks")
    parser.add_argument("--prompt-version", choices=("baseline", "codex_v1"), default="baseline",
                        help="codex_v1 selects and validates passage evidence in one model call")
    args = parser.parse_args(argv)
    if args.k <= 0:
        parser.error("-k must be positive")

    from retrieve import Retriever

    llm = make_llm(args.model)  # fails on a missing key before the embedding model loads
    answer = summarize(args.question, Retriever(args.db).search, llm, args.k,
                       prompt_version=args.prompt_version)
    print_answer(answer, args.show_sources)


if __name__ == "__main__":
    main()
