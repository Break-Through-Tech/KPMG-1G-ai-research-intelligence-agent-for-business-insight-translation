"""Compare two saved summarization prompts, including human scores if available.

  python src/compare_summaries.py --out reports/summarization
"""
import argparse
import csv
import json
from pathlib import Path


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path):
    records = {}
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            if line.strip():
                row = json.loads(line)
                if row["id"] in records:
                    raise SystemExit(f"{path}:{number}: duplicate saved question ID {row['id']!r}")
                records[row["id"]] = row
    return records


def load_reviews(path):
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("reports/summarization"))
    parser.add_argument("--baseline-out", type=Path, help="baseline directory, if separate from --out")
    parser.add_argument("--candidate", choices=("tuned", "tuned_v2", "tuned_v3", "codex_v1"), default="tuned",
                        help="prompt experiment to compare against the baseline")
    args = parser.parse_args(argv)
    baseline_out = args.baseline_out or args.out
    base = load_json(baseline_out / "baseline_summary.json")
    candidate = load_json(args.out / f"{args.candidate}_summary.json")
    for field in ("model", "k", "indexed_unique_papers", "indexed_chunks", "benchmark_questions"):
        if base[field] != candidate[field]:
            raise SystemExit(
                f"Cannot compare runs: different {field} "
                f"({base[field]} vs {candidate[field]})"
            )
    baseline = load_jsonl(baseline_out / "baseline_answers.jsonl")
    modified = load_jsonl(args.out / f"{args.candidate}_answers.jsonl")
    br = load_reviews(baseline_out / "baseline_review.csv")
    tr = load_reviews(args.out / f"{args.candidate}_review.csv")
    shared = sorted(set(baseline) & set(modified))
    for qid in shared:
        a, b = baseline[qid], modified[qid]
        if (a["question"] != b["question"]
                or a["relevant_arxiv_ids"] != b["relevant_arxiv_ids"]
                or a.get("expected_answer", "") != b.get("expected_answer", "")):
            raise SystemExit(f"Cannot compare runs: different benchmark configuration for {qid}")
    rows = []
    for qid in shared:
        a, b = baseline[qid], modified[qid]
        if a["retrieved_arxiv_ids"] != b["retrieved_arxiv_ids"]:
            print(f"NOTE: {qid}: retrieved sources changed; may be model or index variation")
        elif "retrieved_chunks" in a and "retrieved_chunks" in b:
            fields = ("rank", "chunk_id", "arxiv_id", "page_start", "page_end")
            signature = lambda row: [tuple(chunk.get(field) for field in fields)
                                     for chunk in row["retrieved_chunks"]]
            if signature(a) != signature(b):
                print(f"NOTE: {qid}: retrieved chunk order or passages changed despite matching paper IDs")
            elif all("text" in chunk for row in (a, b) for chunk in row["retrieved_chunks"]):
                if [chunk["text"] for chunk in a["retrieved_chunks"]] != [chunk["text"] for chunk in b["retrieved_chunks"]]:
                    print(f"NOTE: {qid}: retrieved passage text changed")
        row = {"id": qid, "question": a["question"],
               "baseline_citation_recall": a["citation_recall"],
               f"{args.candidate}_citation_recall": b["citation_recall"],
               "baseline_citation_precision": a["citation_precision"],
               f"{args.candidate}_citation_precision": b["citation_precision"],
               "baseline_off_target": ";".join(a["off_target_citation_ids"]),
               f"{args.candidate}_off_target": ";".join(b["off_target_citation_ids"]),
               "baseline_accuracy": br.get(qid, {}).get("accuracy_1_to_5", ""),
               f"{args.candidate}_accuracy": tr.get(qid, {}).get("accuracy_1_to_5", ""),
               "baseline_business_usefulness": br.get(qid, {}).get("business_usefulness_1_to_5", ""),
               f"{args.candidate}_business_usefulness":
                   tr.get(qid, {}).get("business_usefulness_1_to_5", "")}
        rows.append(row)
    comparison_path = args.out / (
        "prompt_comparison.csv" if args.candidate == "tuned"
        else f"prompt_comparison_{args.candidate}.csv"
    )
    if rows:
        with open(comparison_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Compared {len(shared)} questions (baseline {len(baseline)}, "
          f"{args.candidate} {len(modified)}).")
    for name, stats in (("baseline", base), (args.candidate, candidate)):
        print(f"{name:8} citation recall={stats['mean_citation_recall']} "
              f"citation precision={stats['mean_citation_precision_for_answers_with_citations']} "
              f"off-target answers={stats['answers_with_off_target_citations']}")
    print(f"Comparison: {comparison_path}")
    print("Review accuracy and usefulness from the scored CSVs; citation IDs alone cannot establish factual correctness.")


if __name__ == "__main__":
    main()
