"""Score retrieval against a benchmark file: Recall@k and MRR at the paper level.

    python src/evaluate_retrieval.py                                   # default benchmark
    python src/evaluate_retrieval.py --benchmark data/benchmark/smoke_questions.jsonl
    python src/evaluate_retrieval.py --out-of-scope data/benchmark/out_of_scope.jsonl

The benchmark is JSONL, one question per line:
    {"id": "q01", "question": "...", "relevant_arxiv_ids": ["2608.20316v1"]}

Retrieval returns chunks. Scoring collapses the ranked chunks into a ranked list of
papers (first appearance wins), because a question is tagged with the paper that
should come back, not a specific chunk.

    Recall@k  share of a question's relevant papers found in the top k papers, averaged
    MRR       1 / rank of the first relevant paper, 0 if none in the pool, averaged

With --out-of-scope, also runs questions the corpus cannot answer and reports how
many fall below the retriever's relevance floor (see retrieve.py), next to the
lowest best-chunk similarity among the in-scope questions. If an in-scope question
ever scores below the floor, the floor is too high for the current index.

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
        row["best_similarity"] = max((getattr(h, "similarity", 0.0) for h in hits), default=0.0)
        rows.append(row)

    n = len(rows)
    summary = {f"recall@{k}": sum(r[f"recall@{k}"] for r in rows) / n for k in ks}
    summary["mrr"] = sum(r["rr"] for r in rows) / n
    summary["questions"] = n
    return rows, summary


def load_out_of_scope(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path} not found.")
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows or any("question" not in r for r in rows):
        raise SystemExit(f"{path} needs one JSON object with a 'question' per line.")
    return rows


def check_out_of_scope(questions: list[dict], search, floor: float | None) -> list[dict]:
    """Best-chunk similarity per question and whether the floor would stop it.

    `search(question)` must return the same chunks summarize.py sends to the LLM,
    because that is where the floor is applied."""
    rows = []
    for q in questions:
        hits = search(q["question"])
        best = max((getattr(h, "similarity", 0.0) for h in hits), default=0.0)
        rows.append(
            {
                "id": q.get("id", ""),
                "question": q["question"],
                "best_similarity": best,
                "below_floor": floor is not None and best < floor,
                "top_papers": ";".join(paper_ranking([h.arxiv_id for h in hits])[:3]),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark", type=Path, default=Path("data/benchmark/questions.jsonl"))
    parser.add_argument("--db", type=Path, default=Path("chroma_db"))
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--pool", type=int, default=50, help="chunks to retrieve before collapsing to papers")
    parser.add_argument("--out", type=Path, default=Path("data/processed/eval"))
    parser.add_argument("--out-of-scope", type=Path, default=None, help="JSONL of questions the corpus cannot answer")
    args = parser.parse_args(argv)
    if min(args.ks) <= 0 or args.pool < max(args.ks):
        parser.error("--ks must be positive and --pool at least max(--ks)")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    from retrieve import Retriever  # imported here so the metric functions stay dependency-free

    benchmark = load_benchmark(args.benchmark)
    retriever = Retriever(args.db)
    pool = min(args.pool, retriever.count())
    rows, summary = evaluate(benchmark, retriever.search, args.ks, pool)

    summary.update(
        benchmark=str(args.benchmark),
        hub_correction=retriever.hub_corrected,
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

    if args.out_of_scope:
        from retrieve import CONTEXT_K, CONTEXT_MAX_PER_PAPER

        floor = retriever.relevance_floor

        def context_search(question):  # the chunks summarize.py would send
            return retriever.search(question, CONTEXT_K, CONTEXT_MAX_PER_PAPER)

        oos = check_out_of_scope(load_out_of_scope(args.out_of_scope), context_search, floor)
        in_scope = check_out_of_scope(benchmark, context_search, floor)
        with open(args.out / f"{args.out_of_scope.stem}_results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(oos[0].keys()))
            writer.writeheader()
            writer.writerows(oos)
        lowest = min(in_scope, key=lambda r: r["best_similarity"])
        lowest_in = lowest["best_similarity"]
        caught = sum(r["below_floor"] for r in oos)
        log.info("%s: %d out-of-scope questions", args.out_of_scope, len(oos))
        if floor is None:
            log.info("  no relevance floor on this index (built without hub correction)")
        else:
            log.info("  relevance floor %.2f catches %d/%d; the rest go to the prompt to turn down", floor, caught, len(oos))
            log.info("  lowest in-scope best similarity %.3f, %s (%s)", lowest_in, lowest["id"],
                     "OK, above floor" if lowest_in >= floor else "BELOW FLOOR, lower it")
        for r in sorted(oos, key=lambda r: r["best_similarity"]):
            log.info("  %s %.3f%s  %s", r["id"], r["best_similarity"], " (caught)" if r["below_floor"] else "", r["question"][:70])


if __name__ == "__main__":
    main()
