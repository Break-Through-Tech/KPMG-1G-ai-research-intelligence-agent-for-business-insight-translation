"""Draw bar charts from the files evaluate_retrieval.py writes.

    python src/evaluate_retrieval.py --benchmark data/benchmark/questions.jsonl
    python src/evaluate_retrieval.py --benchmark data/benchmark/smoke_questions.jsonl
    python src/plot_eval.py                                  # -> docs/figures/

Two charts:
    retrieval_scores.png      Recall@k and MRR for the benchmark, next to the smoke set if it was run
    first_correct_rank.png    for each benchmark question, where the first relevant paper ranked

Rerun after any change to chunking, embeddings or the corpus so the figures match the README.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

RANK_BUCKETS = ["1st", "2nd", "3rd", "4th-5th", "Not in top 5"]

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
BENCHMARK_COLOR = "#2a78d6"
SMOKE_COLOR = "#eb6834"


def rank_bucket(rr: float) -> str:
    """Reciprocal rank -> bucket. rr is 1/rank, or 0 when no relevant paper came back."""
    if rr <= 0:
        return "Not in top 5"
    rank = round(1 / rr)
    if rank <= 3:
        return RANK_BUCKETS[rank - 1]
    return "4th-5th" if rank <= 5 else "Not in top 5"


def count_ranks(rows: list[dict]) -> dict[str, int]:
    counts = {b: 0 for b in RANK_BUCKETS}
    for row in rows:
        counts[rank_bucket(float(row["rr"]))] += 1
    return counts


def metric_names(summary: dict) -> list[str]:
    ks = sorted(int(key.split("@")[1]) for key in summary if key.startswith("recall@"))
    return [f"recall@{k}" for k in ks] + ["mrr"]


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_MUTED, length=0, labelsize=12)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_scores(benchmark: dict, smoke: dict | None, path: Path) -> None:
    import matplotlib.pyplot as plt

    names = metric_names(benchmark)
    labels = ["MRR" if n == "mrr" else n.replace("recall", "Recall") for n in names]
    series = [(f"Benchmark ({benchmark['questions']} questions)", benchmark, BENCHMARK_COLOR)]
    if smoke:
        series.append((f"Smoke test ({smoke['questions']} questions)", smoke, SMOKE_COLOR))

    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    width = 0.72 / len(series)
    for i, (name, summary, color) in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * (width + 0.02)
        values = [summary[n] for n in names]
        bars = ax.bar([x + offset for x in range(len(names))], values, width, color=color, label=name, zorder=3)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.2f}",
                    ha="center", va="bottom", color=TEXT, fontsize=12)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, color=TEXT, fontsize=13)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    fig.legend(loc="upper left", frameon=False, fontsize=11, labelcolor=TEXT_MUTED,
               ncol=len(series), bbox_to_anchor=(0.05, 0.9), handlelength=1.2)
    fig.suptitle("Retrieval scores", x=0.06, ha="left", y=0.97, fontsize=17, color=TEXT, fontweight="bold")
    fig.text(0.06, 0.02, f"Run {benchmark['run_at'][:10]}, top {benchmark['pool']} chunks collapsed to papers.",
             fontsize=9.5, color=TEXT_MUTED)
    plt.subplots_adjust(top=0.8, bottom=0.12, left=0.06, right=0.98)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_ranks(rows: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    counts = count_ranks(rows)
    values = [counts[b] for b in RANK_BUCKETS]
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    bars = ax.bar(range(len(RANK_BUCKETS)), values, 0.55, color=BENCHMARK_COLOR, zorder=3)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.3, str(value),
                ha="center", va="bottom", color=TEXT, fontsize=13)
    ax.set_xticks(range(len(RANK_BUCKETS)))
    ax.set_xticklabels(RANK_BUCKETS, color=TEXT, fontsize=13)
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.set_ylim(0, max(values) + 2)
    ax.set_ylabel("Questions", color=TEXT_MUTED, fontsize=12)
    fig.suptitle("Where the first relevant paper ranked", x=0.06, ha="left", y=0.97,
                 fontsize=17, color=TEXT, fontweight="bold")
    fig.text(0.06, 0.9, f"{len(rows)} benchmark questions", fontsize=12, color=TEXT_MUTED)
    plt.subplots_adjust(top=0.82, bottom=0.1, left=0.08, right=0.98)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-dir", type=Path, default=Path("data/processed/eval"))
    parser.add_argument("--benchmark", default="questions", help="file stem of the benchmark results")
    parser.add_argument("--smoke", default="smoke_questions", help="file stem of the smoke results; skipped if missing")
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = parser.parse_args(argv)

    summary_path = args.eval_dir / f"{args.benchmark}_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"{summary_path} not found. Run python src/evaluate_retrieval.py first.")
    benchmark = json.loads(summary_path.read_text(encoding="utf-8"))
    smoke_path = args.eval_dir / f"{args.smoke}_summary.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8")) if smoke_path.exists() else None
    with open(args.eval_dir / f"{args.benchmark}_results.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    args.out.mkdir(parents=True, exist_ok=True)
    plot_scores(benchmark, smoke, args.out / "retrieval_scores.png")
    plot_ranks(rows, args.out / "first_correct_rank.png")
    print(f"Wrote {args.out / 'retrieval_scores.png'} and {args.out / 'first_correct_rank.png'}")


if __name__ == "__main__":
    main()
