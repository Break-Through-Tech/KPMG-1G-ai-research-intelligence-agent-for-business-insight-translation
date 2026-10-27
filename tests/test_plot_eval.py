"""Chart helper tests. No matplotlib needed: python -m unittest discover -s tests -v"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import plot_eval as pe


class RankBucketTest(unittest.TestCase):
    def test_top_three_ranks(self):
        self.assertEqual(pe.rank_bucket(1.0), "1st")
        self.assertEqual(pe.rank_bucket(0.5), "2nd")
        self.assertEqual(pe.rank_bucket(1 / 3), "3rd")

    def test_fourth_and_fifth_share_a_bucket(self):
        self.assertEqual(pe.rank_bucket(0.25), "4th-5th")
        self.assertEqual(pe.rank_bucket(0.2), "4th-5th")

    def test_below_five_or_missing(self):
        self.assertEqual(pe.rank_bucket(1 / 7), "Not in top 5")
        self.assertEqual(pe.rank_bucket(0.0), "Not in top 5")

    def test_count_ranks_covers_every_row(self):
        rows = [{"rr": "1.0"}, {"rr": "1.0"}, {"rr": "0.5"}, {"rr": "0.0"}]
        counts = pe.count_ranks(rows)
        self.assertEqual(counts, {"1st": 2, "2nd": 1, "3rd": 0, "4th-5th": 0, "Not in top 5": 1})
        self.assertEqual(sum(counts.values()), len(rows))

    def test_metric_names_sorted_by_k_then_mrr(self):
        summary = {"recall@5": 0, "recall@1": 0, "recall@3": 0, "mrr": 0, "questions": 3}
        self.assertEqual(pe.metric_names(summary), ["recall@1", "recall@3", "recall@5", "mrr"])


if __name__ == "__main__":
    unittest.main()
