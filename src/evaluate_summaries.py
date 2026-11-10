"""Evaluate the 26-question RAG summarization benchmark and save comparable runs.

Examples (run from repository root):
  python src/evaluate_summaries.py --check-corpus
  python src/evaluate_summaries.py --version baseline --limit 2 -k 10
  python src/evaluate_summaries.py --version baseline -k 10 --delay 15
  python src/evaluate_summaries.py --version tuned -k 10 --delay 15

No benchmark expected_answer is sent to the LLM. Saved results include source IDs,
answer, citation metrics, and a separate human-scoring sheet.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from evaluate_retrieval import load_benchmark
from evidence import CODEX_V1_PROMPT
from pull_arxiv import base_id
from summarize import DEFAULT_MODEL, SYSTEM_PROMPT, make_llm, summarize


# Keep the baseline unchanged for comparison.
TUNED_PROMPT = """You translate findings from retrieved AI research papers into accurate, useful business insights for a KPMG analyst.

Evidence and citation rules:
- Base every research finding strictly on the provided source chunks; never invent study results, numbers, comparisons, or deployment outcomes.
- Put an [arXiv:ID] citation immediately after EACH sentence containing a factual research claim; use exactly the arXiv ID on its supporting source.
- For claims supported jointly by two papers, cite both IDs. Do not cite a paper that does not actually support the statement.
- For multi-paper questions, synthesize findings across relevant papers rather than only describing the first paper. Do not force irrelevant sources into the answer.
- Preserve reported numbers, study settings, uncertainty, and limitations exactly as supported by the chunks.
- If only part of a question is covered, state the limitation explicitly; if none is covered, reply exactly: The retrieved papers do not answer this.

Communication rules:
- Start by answering the question directly, in plain, precise language.
- Mention the most decision-relevant result, plus its limitation or caveat when available.
- Close with a cautious, actionable business implication grounded in the evidence, citing the source if it depends on a research finding. Do not claim real-world validation unless reported.
- Aim for 3-6 concise sentences; use up to 8 for questions requiring multiple distinct papers.
"""

TUNED_V2_PROMPT = """You translate findings from retrieved AI research papers into accurate, useful business insights for a KPMG analyst.

Evidence and citation rules:
- Answer using only evidence in the retrieved research sources. Do not use outside knowledge or invent findings.
- Cite every factual research claim immediately with the exact format [arXiv:ID], using the ID shown for its supporting source.
- Include the full arXiv ID and version whenever a version is provided, for example [arXiv:2608.20316v1].
- Never use numeric bibliography citations such as [1], [6], or [1, 10].
- Use multiple arXiv citations when multiple papers support a claim, and never cite a paper that was not retrieved.
- Synthesize findings across multiple relevant papers when appropriate; do not force irrelevant papers into the answer.
- Preserve study settings, reported numbers, uncertainty, and limitations. Do not overstate results or invent findings.
- Provide business implications only when relevant and supported by the research evidence.
- If the retrieved evidence is insufficient to answer, say exactly: The retrieved papers do not answer this.

Write a concise answer of 3-6 sentences, in plain language.
"""

TUNED_V3_PROMPT = """You translate findings from retrieved AI research papers into accurate, useful business insights for a KPMG analyst.

Answer relevance:
- Identify what the question specifically asks, and answer that directly using only the retrieved research text.
- Prioritize evidence that directly answers the question. Exclude tangential background, broad related-work discussion, and details that do not help answer it.
- Make every sentence contribute to the answer. Do not expand a narrow question into a broad survey.

Evidence and citations:
- Cite a paper only when its retrieved text directly supports the specific claim beside the citation. Topical similarity or appearing in the retrieved sources is not enough.
- Do not treat a paper's bibliography, citations, or mentions of other work as findings reported by that paper.
- Never invent a finding or citation, and never cite a paper absent from the retrieved sources.
- Include all distinct, directly relevant evidence from retrieved papers when it materially answers the question. Cite each contributing paper for the claim it supports; omit redundant citations and citations added only to appear comprehensive.
- Use the exact format [arXiv:ID], including the full version when provided. Place citations immediately after the claims they support.
- Distinguish reported results from hypotheses and limitations. Preserve important study conditions and uncertainty, avoid unsupported generalizations, and acknowledge conflicting evidence when present.
- If only part of the question is answered, state what evidence is missing. If the retrieved evidence does not answer the question, say exactly: The retrieved papers do not answer this.
- Provide business implications only when directly relevant and supported by the retrieved evidence.

Write a concise but sufficiently detailed answer of 3-6 sentences in plain language.
"""

PROMPTS = {
    "baseline": SYSTEM_PROMPT,
    "tuned": TUNED_PROMPT,
    "tuned_v2": TUNED_V2_PROMPT,
    "tuned_v3": TUNED_V3_PROMPT,
    "codex_v1": CODEX_V1_PROMPT,
}

NUMBERED_SOURCE_HEADER_RE = re.compile(r"(?m)^\[(\d+)\] ID: ([^\n|]+) \|")


class PromptOverrideLLM:
    """Use existing summarize() unchanged, replacing only the system prompt."""
    def __init__(self, llm, prompt: str, clear_source_numbers: bool = False):
        self.llm = llm
        self.prompt = prompt
        self.clear_source_numbers = clear_source_numbers

    def invoke(self, messages):
        messages = list(messages)
        if not messages or messages[0][0] != "system":
            raise ValueError("Unexpected summarizer message structure")
        messages[0] = ("system", self.prompt)
        if self.clear_source_numbers:
            messages = [
                (role, NUMBERED_SOURCE_HEADER_RE.sub(r"Source \1 | arXiv ID: \2 |", content)
                 if role == "human" and isinstance(content, str) else content)
                for role, content in messages
            ]
        return self.llm.invoke(messages)


class RecordingLLM:
    """Capture the exact messages passed to the model, after prompt overrides."""
    def __init__(self, llm):
        self.llm = llm
        self.messages = []

    def invoke(self, messages):
        self.messages = list(messages)
        return self.llm.invoke(self.messages)


def canonical_ids(ids):
    return set(base_id(x) for x in ids)


def metrics(expected_ids, retrieved_ids, cited_ids):
    """Paper-level retrieval and citation metrics, insensitive to arXiv version suffix."""
    expected = canonical_ids(expected_ids)
    retrieved = canonical_ids(retrieved_ids)
    cited = canonical_ids(cited_ids)
    correct = cited & expected
    off_target = cited - expected
    unknown = cited - retrieved
    return {
        "retrieval_recall": round(len(retrieved & expected) / len(expected), 4),
        "citation_recall": round(len(correct) / len(expected), 4),
        "citation_precision": round(len(correct) / len(cited), 4) if cited else None,
        "all_relevant_retrieved": expected.issubset(retrieved),
        "all_relevant_cited": expected.issubset(cited),
        "correct_citation_ids": sorted(correct),
        "off_target_citation_ids": sorted(off_target),
        "unretrieved_citation_ids": sorted(unknown),
        "no_citations": not bool(cited),
    }


def index_collection(db_path: Path):
    import chromadb
    from build_index import DEFAULT_COLLECTION
    client = chromadb.PersistentClient(path=str(db_path))
    try:
        return client.get_collection(DEFAULT_COLLECTION)
    except Exception as exc:
        raise SystemExit(f"No Chroma collection {DEFAULT_COLLECTION!r} in {db_path}. Build the index first.") from exc


def index_corpus_ids(db_path: Path):
    metas = index_collection(db_path).get(include=["metadatas"])["metadatas"]
    return canonical_ids(meta["arxiv_id"] for meta in metas), len(metas)


def index_content_sha256(db_path: Path):
    """Hash indexed passage content and metadata, independent of storage row order."""
    collection = index_collection(db_path)
    data = collection.get(include=["documents", "metadatas"])
    digest = hashlib.sha256()
    digest.update(json.dumps(collection.metadata, sort_keys=True, ensure_ascii=False).encode())
    for row in sorted(zip(data["ids"], data["documents"], data["metadatas"])):
        digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def corpus_coverage(benchmark, index_ids):
    expected = canonical_ids(pid for row in benchmark for pid in row["relevant_arxiv_ids"])
    return expected, sorted(expected - index_ids)


def as_csv_row(record):
    fields = ["id", "question", "expected_answer", "answer", "relevant_arxiv_ids", "retrieved_arxiv_ids",
              "cited_arxiv_ids", "correct_citation_ids", "off_target_citation_ids", "unretrieved_citation_ids",
              "retrieval_recall", "citation_precision", "citation_recall", "all_relevant_retrieved",
              "all_relevant_cited", "no_citations", "prompt_version", "model", "k", "accuracy_1_to_5",
              "business_usefulness_1_to_5", "citation_support_notes"]
    r = {k: record.get(k, "") for k in fields}
    for key in ("relevant_arxiv_ids", "retrieved_arxiv_ids", "cited_arxiv_ids", "correct_citation_ids",
                "off_target_citation_ids", "unretrieved_citation_ids"):
        r[key] = ";".join(record.get(key, []))
    if r["citation_precision"] is None:
        r["citation_precision"] = ""
    return r


def write_review_csv(path, records):
    """Refresh all answer rows without losing existing manually entered scores."""
    existing = {}
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            existing = {row["id"]: row for row in csv.DictReader(f)}
    rows = [as_csv_row(record) for record in records]
    if not rows:
        return
    score_fields = ("accuracy_1_to_5", "business_usefulness_1_to_5", "citation_support_notes")
    for row in rows:
        previous = existing.get(row["id"], {})
        for field in score_fields:
            row[field] = previous.get(field, "")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_previous(path):
    if not path.exists():
        return {}
    records = {}
    with open(path, encoding="utf-8") as f:
        for number, raw in enumerate(f, start=1):
            if raw.strip():
                row = json.loads(raw)
                if row["id"] in records:
                    raise SystemExit(f"{path}:{number}: duplicate saved question ID {row['id']!r}")
                records[row["id"]] = row
    return records


def validate_previous(records, benchmark, *, prompt, version, model, k, index_ids, index_chunks):
    """Reject stale or mixed runs before deciding whether there is work to resume."""
    expected = {row["id"]: row for row in benchmark}
    if len(expected) != len(benchmark):
        raise SystemExit("Benchmark has duplicate question IDs.")
    unknown = set(records) - set(expected)
    if unknown:
        raise SystemExit("Saved run contains unknown question IDs: " + ", ".join(sorted(unknown)))
    config = {"k": k, "model": model, "index_chunks": index_chunks,
              "indexed_unique_papers": len(index_ids), "prompt_version": version,
              "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    for qid, record in records.items():
        question = expected[qid]
        for field in ("question", "relevant_arxiv_ids", "expected_answer"):
            default = "" if field == "expected_answer" else None
            if record.get(field, default) != question.get(field, default):
                raise SystemExit(f"Saved run has a different benchmark {field} for {qid}; use a new --out directory.")
        for field, value in config.items():
            if record.get(field) != value:
                raise SystemExit(f"Saved run has a different {field} for {qid}; use a new --out directory.")


def build_run_manifest(benchmark_path, *, prompt, model, k, index_ids, index_chunks, index_sha256):
    """Freeze new experiment inputs without storing credentials or environment values."""
    source_dir = Path(__file__).resolve().parent
    source_hashes = {
        name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
        for name in ("summarize.py", "evidence.py", "evaluate_summaries.py", "retrieve.py", "build_index.py")
    }
    return {
        "schema_version": 1, "prompt_version": "codex_v1", "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(benchmark_path).read_bytes()).hexdigest(),
        "model": model, "k": k, "indexed_arxiv_ids": sorted(index_ids),
        "indexed_chunks": index_chunks, "index_content_sha256": index_sha256,
        "source_sha256": source_hashes,
    }


def preserve_run_manifest(path, manifest, *, has_previous=False):
    """An existing manifest is immutable; refinements require a separate output directory."""
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise SystemExit("Existing codex_v1 manifest differs from current inputs or implementation; use a new --out directory.")
    elif has_previous:
        raise SystemExit("Existing codex_v1 answers have no run manifest; use a new --out directory.")
    else:
        with path.open("x", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
            f.write("\n")


def summarize_results(records, benchmark_count, prompt, model, k, index_ids, index_chunks):
    rows = list(records.values())
    def mean(key, skip_none=False):
        vals = [r[key] for r in rows if not skip_none or r[key] is not None]
        return round(sum(vals) / len(vals), 4) if vals else None
    return {
        "completed": len(rows), "benchmark_questions": benchmark_count,
        "all_26_complete": len(rows) == benchmark_count,
        "mean_retrieval_recall": mean("retrieval_recall"),
        "mean_citation_recall": mean("citation_recall"),
        "mean_citation_precision_for_answers_with_citations": mean("citation_precision", skip_none=True),
        "answers_with_no_citations": sum(r["no_citations"] for r in rows),
        "answers_with_unretrieved_citations": sum(bool(r["unretrieved_citation_ids"]) for r in rows),
        "answers_with_off_target_citations": sum(bool(r["off_target_citation_ids"]) for r in rows),
        "all_relevant_retrieved_count": sum(r["all_relevant_retrieved"] for r in rows),
        "all_relevant_cited_count": sum(r["all_relevant_cited"] for r in rows),
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "model": model, "k": k,
        "indexed_unique_papers": len(index_ids), "indexed_chunks": index_chunks,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": "Citation metrics compare arXiv IDs, not whether an individual claim is actually supported. Use the review CSV to spot-check factual grounding, accuracy, and business usefulness."
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark", type=Path, default=Path("data/benchmark/questions.jsonl"))
    parser.add_argument("--db", type=Path, default=Path("chroma_db"))
    parser.add_argument("--out", type=Path, default=Path("reports/summarization"))
    parser.add_argument("--version", choices=tuple(PROMPTS), default="baseline")
    parser.add_argument("--model", default=None, help="GEMINI_MODEL in .env if omitted")
    parser.add_argument("-k", type=int, default=10, help="number of chunks per question, same for both prompts")
    parser.add_argument("--limit", type=int, help="run only first N questions; resumes on next invocation")
    parser.add_argument("--delay", type=float, default=15.0, help="seconds between requests for free-tier quotas")
    parser.add_argument("--check-corpus", action="store_true", help="validate indexed papers without making API calls")
    parser.add_argument("--allow-partial-index", action="store_true", help="override missing-papers safety guard for experiments only")
    parser.add_argument("--restart", action="store_true", help="delete this prompt version's saved results and review CSV")
    args = parser.parse_args(argv)
    if args.k <= 0 or args.delay < 0 or (args.limit is not None and args.limit <= 0):
        parser.error("-k and --limit must be positive; --delay must be nonnegative")

    from dotenv import load_dotenv
    load_dotenv()
    benchmark = load_benchmark(args.benchmark)
    index_ids, index_chunks = index_corpus_ids(args.db)
    expected_ids, missing = corpus_coverage(benchmark, index_ids)
    print(f"Index: {index_chunks} chunks from {len(index_ids)} papers; benchmark: {len(benchmark)} questions / {len(expected_ids)} expected papers")
    if missing:
        print("MISSING benchmark papers from index: " + ", ".join(missing))
    if args.check_corpus:
        if missing:
            raise SystemExit(2)
        print("All benchmark papers are present in the index.")
        return
    if missing and not args.allow_partial_index:
        raise SystemExit("Stopped: index incomplete. Load the fixed 50-paper corpus first (or use --allow-partial-index for a NON-OFFICIAL smoke run).")

    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / f"{args.version}_answers.jsonl"
    review_path = args.out / f"{args.version}_review.csv"
    summary_path = args.out / f"{args.version}_summary.json"
    manifest_path = args.out / f"{args.version}_manifest.json"
    if args.restart:
        if args.version == "codex_v1" and any(p.exists() for p in (results_path, review_path, summary_path, manifest_path)):
            raise SystemExit("codex_v1 experiment files are preserved; use a new --out directory for another run.")
        for p in (results_path, review_path, summary_path):
            p.unlink(missing_ok=True)
    previous = load_previous(results_path)
    prompt = PROMPTS[args.version]
    model_name = args.model or os.getenv("GEMINI_MODEL") or DEFAULT_MODEL
    validate_previous(previous, benchmark, prompt=prompt, version=args.version, model=model_name,
                      k=args.k, index_ids=index_ids, index_chunks=index_chunks)
    if args.version == "codex_v1":
        manifest = build_run_manifest(args.benchmark, prompt=prompt, model=model_name, k=args.k,
                                      index_ids=index_ids, index_chunks=index_chunks,
                                      index_sha256=index_content_sha256(args.db))
        preserve_run_manifest(manifest_path, manifest, has_previous=bool(previous))
    if set(previous) == {row["id"] for row in benchmark}:
        print("All benchmark questions already saved; no API calls needed.")
    else:
        from retrieve import Retriever
        recorder = RecordingLLM(make_llm(args.model))
        llm = recorder if args.version == "codex_v1" else PromptOverrideLLM(
            recorder, prompt, clear_source_numbers=args.version in ("tuned_v2", "tuned_v3"))
        retriever = Retriever(args.db)
        questions = benchmark[:args.limit] if args.limit else benchmark
        for row in questions:
            qid = row["id"]
            if qid in previous:
                print(f"SKIP {qid} (saved)")
                continue
            try:
                recorder.messages = []
                options = {"prompt_version": "codex_v1"} if args.version == "codex_v1" else {}
                ans = summarize(row["question"], retriever.search, llm, args.k, **options)
            except Exception as exc:
                raise SystemExit(f"STOP at {qid}: {type(exc).__name__}: {exc}\nPreviously completed answers are saved; rerun the same command to resume.") from exc
            scores = metrics(row["relevant_arxiv_ids"], ans.retrieved_ids, ans.cited_ids)
            record = {
                "id": qid, "question": row["question"], "expected_answer": row.get("expected_answer", ""),
                "answer": ans.text,
                "relevant_arxiv_ids": row["relevant_arxiv_ids"],
                "retrieved_arxiv_ids": ans.retrieved_ids, "cited_arxiv_ids": ans.cited_ids,
                "retrieved_chunks": [
                    {"rank": h.rank, "chunk_id": h.chunk_id, "arxiv_id": h.arxiv_id,
                     "score": round(h.score, 4), "page_start": h.page_start, "page_end": h.page_end}
                    for h in ans.hits
                ],
                "model_input_messages": recorder.messages,
                **scores,
                "prompt_version": args.version, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "model": model_name, "k": args.k,
                "indexed_unique_papers": len(index_ids), "index_chunks": index_chunks,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            for chunk, hit in zip(record["retrieved_chunks"], ans.hits):
                chunk.update(text=hit.text, title=hit.title)
            if args.version == "codex_v1":
                record.update(raw_response=ans.raw_response, evidence=ans.evidence,
                              validation_errors=ans.validation_errors)
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            previous[qid] = record
            print(f"{qid}: retrieved {scores['retrieval_recall']:.0%} of expected IDs; "
                  f"cited {scores['citation_recall']:.0%}; "
                  f"off-target {len(scores['off_target_citation_ids'])}; "
                  f"unretrieved {len(scores['unretrieved_citation_ids'])}")
            if args.delay and row is not questions[-1]:
                time.sleep(args.delay)

    # Results are based on the original benchmark ordering, not completion order.
    ordered = {r["id"]: previous[r["id"]] for r in benchmark if r["id"] in previous}
    summary = summarize_results(ordered, len(benchmark), prompt, model_name, args.k, index_ids, index_chunks)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_review_csv(review_path, list(ordered.values()))
    print(f"Saved: {results_path}\nSummary: {summary_path}\nReview sheet: {review_path}")
    print(f"Completed: {summary['completed']}/{summary['benchmark_questions']} | "
          f"retrieval recall: {summary['mean_retrieval_recall']} | citation recall: {summary['mean_citation_recall']}")
    if summary["completed"] < len(benchmark):
        print("Partial run: repeat with no --limit to finish. Existing results are skipped.")


if __name__ == "__main__":
    main()
