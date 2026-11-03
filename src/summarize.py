"""Answer a question from the retrieved chunks, citing the arXiv id behind each claim.

    python src/summarize.py "How close are AI agents to designing their own training algorithms?"
    python src/summarize.py "your question" -k 8 --model gemini-3.6-flash --show-sources
    python src/summarize.py --batch data/benchmark/questions.jsonl --prompt v1   # old prompt, for comparison
    python src/summarize.py --batch data/benchmark/questions.jsonl --prompt v2

Retrieves the top-k chunks with retrieve.py (at most --max-per-paper from any one
paper, so multi-paper questions get several papers), sends them to a free-tier
Gemini model through LangChain, and prints an answer where every claim ends in
[arXiv:ID]. Any citation to a paper that was not retrieved is flagged.

Two layers decide when not to answer:
  1. Retrieval: if even the best chunk is below the retriever's relevance floor,
     the question is clearly outside the corpus and no LLM call is made.
  2. Prompt: the model is told to say when the sources do not cover the question,
     only partly cover it, or contradict its premise. Most out-of-scope questions
     land here, because similarity scores alone cannot separate them (see
     docs/eval/retrieval_tuning_nov.md).

--batch writes one JSON line per question to data/processed/answers/, with empty
rubric_accuracy and rubric_usefulness fields to score by hand against
data/benchmark/README.md.

Needs GOOGLE_API_KEY in .env (see .env.example). Free-tier limits vary by project,
so check yours in Google AI Studio before running a big batch.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from evaluate_retrieval import paper_ranking
from pull_arxiv import base_id
from retrieve import CONTEXT_K, CONTEXT_MAX_PER_PAPER

DEFAULT_MODEL = "gemini-3.5-flash-lite"  # the lightest Gemini model; override with --model or GEMINI_MODEL
DEFAULT_K = CONTEXT_K  # 8 chunks, at most 2 per paper: see retrieve.py
DEFAULT_MAX_PER_PAPER = CONTEXT_MAX_PER_PAPER
MAX_CHARS_PER_CHUNK = 1500  # safety cap; normal 120-word chunks are well under it

NOT_COVERED = "The retrieved papers do not cover this question."

ARXIV_ID_RE = re.compile(r"\d{4}\.\d{4,5}(?:v\d+)?")
# Any [...] holding an arXiv id: [arXiv:A], [A], [arXiv:A, arXiv:B], [arXiv: A; B]
CITATION_BRACKET_RE = re.compile(r"\[[^\[\]]*\d{4}\.\d{4,5}[^\[\]]*\]")

# v1 is the October prompt, kept so the two can be scored side by side with --batch.
PROMPT_V1 = """You answer questions about AI research for a business audience.

Rules:
- Use ONLY the numbered sources in the user message. Do not use outside knowledge.
- End every sentence that states a finding with its citation in the form [arXiv:ID], using the ID shown for that source.
- If two sources support a claim, cite both: [arXiv:ID1] [arXiv:ID2].
- Keep the paper's own numbers and names exactly as written. Do not round or rename.
- If the sources do not answer the question, say "The retrieved papers do not answer this." and stop.
- Plain language, 3 to 6 sentences."""

# v2 follows the answer pattern and stress tests in the Challenge Advisor's
# Personas_Prompts/Break_Through_Tech_2026_Sample_User_Prompts.docx.
PROMPT_V2 = f"""You are a research analyst at a professional-services firm. You explain AI research to business readers, using only the numbered source passages in the user message.

Step 1. Decide how well the sources answer the question.
- If no source addresses it, make the first line of your reply exactly "{NOT_COVERED}" with no heading or label before it. On the next line, name the closest topic the sources do cover in one sentence. Stop there.
- If the sources only partly address it, do not use that sentence. Say what they cover and what is missing, then answer.
- If the question assumes something the sources contradict, correct the assumption first.
- If the question asks for a conclusion the sources do not support, say so instead of giving it.
- If a term in the question is ambiguous, state the reading you used.

Step 2. Answer in this format:

**Answer:** 2-3 sentences that directly answer the question.

**Evidence:** one bullet per paper you used: what it did and found, with its own numbers, ending with [arXiv:ID].

**Limitations:** what the evidence does not show, for example lab or benchmark-only results, one study, small scale, or a narrow setting.

**Business implications:** 2-3 bullets on what this could mean for an organization and who it affects. These are interpretation, not findings, so word them as possibilities.

Rules:
- Cite every finding as [arXiv:ID] with the ID shown for its source. If two sources support a claim, cite both.
- Keep numbers, names and units exactly as the paper states them. Do not round or rename.
- Report what a paper claims as the authors' result, not as settled fact.
- Do not count two passages from the same paper as independent evidence.
- Plain language. Explain any technical term in a few words the first time you use it.
- Use no outside knowledge."""

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}


@dataclass
class Answer:
    question: str
    text: str
    hits: list = field(default_factory=list)
    retrieved_ids: list[str] = field(default_factory=list)
    cited_ids: list[str] = field(default_factory=list)
    uncited_sources: list[str] = field(default_factory=list)  # retrieved but never cited
    unknown_citations: list[str] = field(default_factory=list)  # cited but never retrieved
    abstained: bool = False  # True when the answer says the papers do not cover the question
    stopped_at_retrieval: bool = False  # True when the relevance floor stopped it before the LLM


def load_paper_info(path: Path = Path("data/metadata.csv")) -> dict[str, dict]:
    """arxiv_id -> {first_author, published}. Empty when the metadata file is missing."""
    if not path.exists():
        return {}
    info = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            authors = [a.strip() for a in row.get("authors", "").split(";") if a.strip()]
            first = authors[0] if authors else ""
            if len(authors) > 1:
                first += " et al."
            info[row["arxiv_id"]] = {"first_author": first, "published": row.get("published", "")}
    return info


def format_sources(hits: Sequence, max_chars: int = MAX_CHARS_PER_CHUNK, paper_info: dict | None = None) -> str:
    paper_info = paper_info or {}
    blocks = []
    for i, hit in enumerate(hits, start=1):
        text = " ".join(hit.text.split())
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + " ..."
        extra = paper_info.get(hit.arxiv_id, {})
        byline = " | ".join(x for x in (extra.get("first_author"), extra.get("published")) if x)
        header = f"[{i}] ID: {hit.arxiv_id} | {hit.title}" + (f" | {byline}" if byline else "")
        blocks.append(f"{header} | pages {hit.page_start}-{hit.page_end}\n{text}")
    return "\n\n".join(blocks)


def build_messages(question: str, hits: Sequence, prompt: str = "v2", paper_info: dict | None = None) -> list[tuple[str, str]]:
    """(role, content) pairs. LangChain chat models accept these directly."""
    human = f"Sources:\n\n{format_sources(hits, paper_info=paper_info)}\n\nQuestion: {question}"
    return [("system", PROMPTS[prompt]), ("human", human)]


def extract_citations(text: str) -> list[str]:
    """arXiv ids cited inside [...] brackets, in order of first appearance.
    An id mentioned outside a bracket is not counted as a citation."""
    ids: list[str] = []
    for bracket in CITATION_BRACKET_RE.findall(text):
        for cid in ARXIV_ID_RE.findall(bracket):
            if cid not in ids:
                ids.append(cid)
    return ids


DECLINE_SENTENCES = (NOT_COVERED.lower(), "the retrieved papers do not answer this.")  # v2, v1
LEADING_LABEL_RE = re.compile(r"""^(?:[#>*_\s"'\u201c]|answer\s*:)+""", re.IGNORECASE)


def is_abstention(text: str) -> bool:
    """True when the reply's first line is the decline sentence of either prompt.

    Markdown, quotes and an "Answer:" label in front of it are ignored. A sentence
    that only starts the same way ("...do not cover this question directly, but")
    is partial coverage, not a decline, so the full sentence with its period must match.
    """
    first_line = next((ln for ln in text.strip().splitlines() if ln.strip()), "")
    head = LEADING_LABEL_RE.sub("", first_line).lower()
    return any(head.startswith(sentence) for sentence in DECLINE_SENTENCES)


def not_covered_answer(question: str, hits: Sequence) -> Answer:
    nearest = paper_ranking([h.arxiv_id for h in hits])[:3]
    titles = {h.arxiv_id: h.title for h in hits}
    lines = [NOT_COVERED, "None of the retrieved passages is close to the question. Nearest papers in the corpus:"]
    lines += [f"  {pid}  {titles.get(pid, '')}" for pid in nearest]
    return Answer(
        question=question,
        text="\n".join(lines),
        hits=list(hits),
        retrieved_ids=paper_ranking([h.arxiv_id for h in hits]),
        abstained=True,
        stopped_at_retrieval=True,
    )


def summarize(
    question: str,
    search: Callable[..., Sequence],
    llm,
    k: int = DEFAULT_K,
    max_per_paper: int | None = DEFAULT_MAX_PER_PAPER,
    relevance_floor: float | None = None,
    prompt: str = "v2",
    paper_info: dict | None = None,
) -> Answer:
    """search(question, k, max_per_paper=...) returns hits like retrieve.Retriever.search does.
    llm is any LangChain chat model. relevance_floor=None skips the retrieval-level check."""
    hits = list(search(question, k, max_per_paper=max_per_paper))
    if not hits:
        return Answer(question=question, text="No chunks were retrieved, so there is nothing to summarize.")
    if relevance_floor is not None and max(getattr(h, "similarity", 0.0) for h in hits) < relevance_floor:
        return not_covered_answer(question, hits)

    retrieved = paper_ranking([h.arxiv_id for h in hits])
    text = llm.invoke(build_messages(question, hits, prompt, paper_info)).text.strip()
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
        abstained=is_abstention(text),
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


def print_answer(answer: Answer, show_sources: bool = False, paper_info: dict | None = None) -> None:
    print(answer.text)
    if answer.stopped_at_retrieval:
        return
    print()
    paper_info = paper_info or {}
    titles = {h.arxiv_id: h.title for h in answer.hits}
    print("Sources:")
    for pid in answer.retrieved_ids:
        mark = "not cited" if pid in answer.uncited_sources else "cited"
        extra = paper_info.get(pid, {})
        byline = ", ".join(x for x in (extra.get("first_author"), extra.get("published")) if x)
        print(f"  {pid}  ({mark})  {titles.get(pid, '')}" + (f". {byline}" if byline else ""))
        print(f"      https://arxiv.org/abs/{pid}")
    if answer.unknown_citations:
        print(f"\nWARNING: cites papers that were not retrieved: {', '.join(answer.unknown_citations)}")
    if show_sources:
        print()
        for hit in answer.hits:
            print(f"{hit.rank}. [{hit.score:.3f}] {hit.arxiv_id} p.{hit.page_start}-{hit.page_end}")
            print(f"   {hit.text[:300]}...")


def run_batch(path: Path, answer_one: Callable[[str], Answer], out_dir: Path, prompt: str, pause: float) -> Path:
    """Answer every question in a JSONL file and write one JSON line per answer, ready for rubric scoring."""
    with open(path, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}_{prompt}.jsonl"
    with open(out_path, "w", encoding="utf-8") as out:
        for i, q in enumerate(questions):
            if i and pause:
                time.sleep(pause)  # stay under free-tier requests-per-minute limits
            try:
                a = answer_one(q["question"])
                error = ""
            except Exception as exc:  # a rate limit or API error on one question should not end the run
                a = Answer(question=q["question"], text="")
                error = f"{exc.__class__.__name__}: {exc}"[:300]
            relevant = q.get("relevant_arxiv_ids", [])
            row = {
                "id": q.get("id", ""),
                "prompt": prompt,
                "question": q["question"],
                "expected_answer": q.get("expected_answer", ""),
                "answer": a.text,
                "retrieved": a.retrieved_ids,
                "cited": a.cited_ids,
                "relevant_cited": sorted({base_id(c) for c in a.cited_ids} & {base_id(r) for r in relevant}),
                "unknown_citations": a.unknown_citations,
                "abstained": a.abstained,
                "stopped_at_retrieval": a.stopped_at_retrieval,
                "error": error,
                "rubric_accuracy": None,
                "rubric_usefulness": None,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            status = f"ERROR {error}" if error else ("abstained" if a.abstained else f"{len(a.cited_ids)} papers cited")
            print(f"{row['id']}: {status}")
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question", nargs="?")
    parser.add_argument("-k", type=int, default=DEFAULT_K)
    parser.add_argument("--max-per-paper", type=int, default=DEFAULT_MAX_PER_PAPER, help="0 for no cap")
    parser.add_argument("--db", type=Path, default=Path("chroma_db"))
    parser.add_argument("--model", default=None, help=f"default: GEMINI_MODEL from .env, else {DEFAULT_MODEL}")
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="v2", help="v1 is the October prompt")
    parser.add_argument("--show-sources", action="store_true", help="also print the retrieved chunks")
    parser.add_argument("--batch", type=Path, default=None, help="JSONL of questions to answer in one run")
    parser.add_argument("--out", type=Path, default=Path("data/processed/answers"))
    parser.add_argument("--pause", type=float, default=4.0, help="seconds between batch calls")
    args = parser.parse_args(argv)
    if args.k <= 0:
        parser.error("-k must be positive")
    if args.max_per_paper < 0:
        parser.error("--max-per-paper must be 0 or more")
    if bool(args.question) == bool(args.batch):
        parser.error("give either a question or --batch FILE")

    from retrieve import Retriever  # loads chromadb and the embedding model, so only when running

    llm = make_llm(args.model)  # fails on a missing key before the embedding model loads
    retriever = Retriever(args.db)
    paper_info = load_paper_info()

    def answer_one(question: str) -> Answer:
        return summarize(
            question,
            retriever.search,
            llm,
            k=args.k,
            max_per_paper=args.max_per_paper or None,
            relevance_floor=retriever.relevance_floor,
            prompt=args.prompt,
            paper_info=paper_info,
        )

    if args.batch:
        out_path = run_batch(args.batch, answer_one, args.out, args.prompt, args.pause)
        print(f"Wrote {out_path}")
    else:
        print_answer(answer_one(args.question), args.show_sources, paper_info)


if __name__ == "__main__":
    main()
