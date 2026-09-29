"""Score retrieval against a benchmark file: Recall@k and MRR at the paper level.

    python src/evaluate_retrieval.py                                   # default benchmark
    python src/evaluate_retrieval.py --benchmark data/benchmark/smoke_questions.jsonl

The benchmark is JSONL, one question per line:
    {"id": "q01", "question": "...", "relevant_arxiv_ids": ["2608.20316v1"]}

Retrieval returns chunks. Scoring collapses the ranked chunks into a ranked list of
papers (first appearance wins), because a question is tagged with the paper that
should come back, not a specific chunk.

    Recall@k  share of a question's relevant papers found in the top k papers, averaged
    MRR       1 / rank of the first relevant paper, 0 if none in the pool, averaged

Writes per-question rows and a summary to data/processed/eval/.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("evaluate_retrieval")


def load_benchmark(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Benchmark {path} not found. See data/benchmark/README.md for the format.")
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = {"id", "question", "relevant_arxiv_ids"} - row.keys()
            if missing or not row["relevant_arxiv_ids"]:
                raise SystemExit(f"{path}:{n} is missing {sorted(missing) or 'relevant_arxiv_ids values'}")
            rows.append(row)
    if not rows:
        raise SystemExit(f"{path} has no questions.")
    return rows


def paper_ranking(chunk_arxiv_ids: list[str]) -> list[str]:
    """Ranked chunk hits -> ranked unique papers, keeping first-appearance order."""
    seen, ranked = set(), []
    for arxiv_id in chunk_arxiv_ids:
        if arxiv_id not in seen:
            seen.add(arxiv_id)
            ranked.append(arxiv_id)
    return ranked


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    return next((1 / i for i, pid in enumerate(ranked, start=1) if pid in relevant), 0.0)


def evaluate(benchmark: list[dict], search, ks: list[int], pool: int) -> tuple[list[dict], dict]:
    """`search(question, n)` returns ranked chunk hits with an `.arxiv_id`."""
    rows = []
    for q in benchmark:
        relevant = set(q["relevant_arxiv_ids"])
        hits = search(q["question"], pool)
        ranked = paper_ranking([h.arxiv_id for h in hits])
        row = {"id": q["id"], "question": q["question"], "relevant": ";".join(sorted(relevant))}
        for k in ks:
            row[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
        row["rr"] = reciprocal_rank(ranked, relevant)
        row["top_papers"] = ";".join(ranked[: max(ks)])
        row["top_chunk"] = hits[0].chunk_id if hits else ""
        rows.append(row)

    n = len(rows)
    summary = {f"recall@{k}": sum(r[f"recall@{k}"] for r in rows) / n for k in ks}
    summary["mrr"] = sum(r["rr"] for r in rows) / n
    summary["questions"] = n
    return rows, summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark", type=Path, default=Path("data/benchmark/questions.jsonl"))
    parser.add_argument("--db", type=Path, default=Path("chroma_db"))
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--pool", type=int, default=50, help="chunks to retrieve before collapsing to papers")
    parser.add_argument("--out", type=Path, default=Path("data/processed/eval"))
    args = parser.parse_args(argv)
    if min(args.ks) <= 0 or args.pool < max(args.ks):
        parser.error("--ks must be positive and --pool at least max(--ks)")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    from retrieve import Retriever  # imported here so the metric functions stay dependency-free

    benchmark = load_benchmark(args.benchmark)
    retriever = Retriever(args.db)
    pool = min(args.pool, retriever.collection.count())
    rows, summary = evaluate(benchmark, retriever.search, args.ks, pool)

    summary.update(
        benchmark=str(args.benchmark),
        pool=pool,
        run_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.benchmark.stem
    with open(args.out / f"{stem}_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(args.out / f"{stem}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log.info("%s: %d questions", args.benchmark, summary["questions"])
    for k in args.ks:
        log.info("  Recall@%d  %.3f", k, summary[f"recall@{k}"])
    log.info("  MRR       %.3f", summary["mrr"])
    for r in rows:
        if r["rr"] < 1:
            log.info("  miss or late: %s (rr=%.2f, got %s)", r["id"], r["rr"], r["top_papers"])


if __name__ == "__main__":
    main()
