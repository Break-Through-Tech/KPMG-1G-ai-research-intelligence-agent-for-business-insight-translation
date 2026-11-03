"""Hub correction, per-paper cap and relevance floor. No model or database needed:
python -m unittest discover -s tests -v"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import build_index as bi
import evaluate_retrieval as ev
import retrieve as rt


def hit(arxiv_id, score, similarity=None):
    return rt.Hit(rank=0, chunk_id=f"{arxiv_id}::{score}", arxiv_id=arxiv_id, title="t", page_start=1, page_end=1,
                  score=score, text="x", similarity=score if similarity is None else similarity)


class CenterTest(unittest.TestCase):
    def test_rows_are_unit_length_and_the_shared_direction_is_gone(self):
        vectors = np.array([[1.0, 0.2], [1.0, -0.2], [1.0, 0.0]])
        centered = bi.center(vectors, vectors.mean(axis=0))
        np.testing.assert_allclose(np.linalg.norm(centered[:2], axis=1), [1.0, 1.0], atol=1e-6)
        # Before centering the first two rows are near-duplicates; after, they point opposite ways.
        self.assertGreater(vectors[0] @ vectors[1] / (np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1])), 0.9)
        self.assertLess(centered[0] @ centered[1], -0.99)

    def test_zero_vector_does_not_divide_by_zero(self):
        out = bi.center(np.array([[1.0, 1.0]]), np.array([1.0, 1.0]))
        self.assertTrue(np.all(np.isfinite(out)))


class HubScoreTest(unittest.TestCase):
    def test_chunk_close_to_every_reference_gets_the_highest_hub_score(self):
        chunks = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
        references = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8], [0.8, 0.6]])
        hubs = bi.hub_scores(chunks, references, k=3)
        self.assertEqual(int(np.argmax(hubs)), 2)  # the "in between" chunk is near all of them

    def test_k_larger_than_reference_set_is_clamped(self):
        hubs = bi.hub_scores(np.eye(2), np.eye(2), k=50)
        np.testing.assert_allclose(hubs, [0.5, 0.5])

    def test_reference_file_skips_comments_and_blanks(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("# header\n\nfirst question\n  # indented comment\nsecond question\n")
        try:
            self.assertEqual(bi.load_reference_queries(Path(f.name)), ["first question", "second question"])
        finally:
            Path(f.name).unlink()

    def test_reference_queries_do_not_leak_the_benchmark(self):
        root = Path(__file__).resolve().parents[1]
        references = set(bi.load_reference_queries(root / "data" / "reference_queries.txt"))
        benchmark = {q["question"] for q in ev.load_benchmark(root / "data" / "benchmark" / "questions.jsonl")}
        self.assertEqual(references & benchmark, set())


class SelectHitsTest(unittest.TestCase):
    def test_sorts_by_score_and_reranks_from_one(self):
        out = rt.select_hits([hit("a", 0.1), hit("b", 0.9), hit("c", 0.5)], k=2)
        self.assertEqual([(h.arxiv_id, h.rank) for h in out], [("b", 1), ("c", 2)])

    def test_per_paper_cap_lets_more_papers_in(self):
        hits = [hit("a", 0.9), hit("a", 0.8), hit("a", 0.7), hit("b", 0.6), hit("c", 0.5)]
        self.assertEqual([h.arxiv_id for h in rt.select_hits(hits, k=3)], ["a", "a", "a"])
        self.assertEqual([h.arxiv_id for h in rt.select_hits(hits, k=3, max_per_paper=2)], ["a", "a", "b"])
        self.assertEqual([h.arxiv_id for h in rt.select_hits(hits, k=3, max_per_paper=1)], ["a", "b", "c"])

    def test_returns_fewer_than_k_when_the_cap_runs_out(self):
        self.assertEqual(len(rt.select_hits([hit("a", 0.9), hit("a", 0.8)], k=5, max_per_paper=1)), 1)

    def test_hub_penalty_can_reorder(self):
        # Same similarity; the chunk with the higher hub score should drop below the other.
        hub_heavy = hit("hub", 2 * 0.6 - 0.5, similarity=0.6)
        specific = hit("specific", 2 * 0.6 - 0.2, similarity=0.6)
        self.assertEqual(rt.select_hits([hub_heavy, specific], k=2)[0].arxiv_id, "specific")


class RelevanceFloorTest(unittest.TestCase):
    def test_best_similarity(self):
        self.assertEqual(rt.best_similarity([hit("a", 0.9, 0.31), hit("b", 0.2, 0.45)]), 0.45)
        self.assertEqual(rt.best_similarity([]), float("-inf"))

    def test_out_of_scope_check_flags_only_questions_below_the_floor(self):
        scores = {"far away": 0.30, "close": 0.50}

        def search(question):
            return [hit("p1", scores[question])]

        rows = ev.check_out_of_scope([{"id": "o1", "question": "far away"}, {"id": "o2", "question": "close"}], search, 0.38)
        self.assertEqual([r["below_floor"] for r in rows], [True, False])
        rows = ev.check_out_of_scope([{"question": "far away"}], search, None)
        self.assertFalse(rows[0]["below_floor"])


if __name__ == "__main__":
    unittest.main()
