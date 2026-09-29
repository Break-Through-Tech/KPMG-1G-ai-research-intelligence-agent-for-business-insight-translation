"""Metric tests. No model or database needed: python -m unittest discover -s tests -v"""

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import evaluate_retrieval as ev


@dataclass
class FakeHit:
    arxiv_id: str
    chunk_id: str = "x"


class MetricTest(unittest.TestCase):
    def test_paper_ranking_keeps_first_appearance(self):
        self.assertEqual(ev.paper_ranking(["a", "a", "b", "a", "c", "b"]), ["a", "b", "c"])

    def test_recall_at_k(self):
        ranked = ["a", "b", "c"]
        self.assertEqual(ev.recall_at_k(ranked, {"b"}, 1), 0.0)
        self.assertEqual(ev.recall_at_k(ranked, {"b"}, 2), 1.0)
        self.assertEqual(ev.recall_at_k(ranked, {"a", "c"}, 2), 0.5)

    def test_reciprocal_rank(self):
        self.assertEqual(ev.reciprocal_rank(["a", "b", "c"], {"c"}), 1 / 3)
        self.assertEqual(ev.reciprocal_rank(["a", "b"], {"z"}), 0.0)

    def test_evaluate_averages(self):
        bench = [
            {"id": "q1", "question": "one", "relevant_arxiv_ids": ["a"]},
            {"id": "q2", "question": "two", "relevant_arxiv_ids": ["b"]},
        ]
        results = {"one": ["a", "a", "b"], "two": ["a", "c", "b"]}

        def search(question, n):
            return [FakeHit(pid) for pid in results[question][:n]]

        rows, summary = ev.evaluate(bench, search, ks=[1, 3], pool=10)
        self.assertEqual(summary["recall@1"], 0.5)
        self.assertEqual(summary["recall@3"], 1.0)
        self.assertAlmostEqual(summary["mrr"], (1 + 1 / 3) / 2)
        self.assertEqual(rows[1]["top_papers"], "a;c;b")


if __name__ == "__main__":
    unittest.main()
