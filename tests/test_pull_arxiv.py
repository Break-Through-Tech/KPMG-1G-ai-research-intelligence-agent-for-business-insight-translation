"""Offline tests for the arXiv pull helpers (no network)."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import parse_and_chunk as pc
import pull_arxiv as pa


class FilenameTest(unittest.TestCase):
    def test_safe_filename_round_trips_through_the_parser(self):
        name = pa.safe_filename("2608.20316v1", "Pandora's AI Model Routing Box: Efficient\nAllocation")
        self.assertEqual(name, "2608.20316v1_Pandora's AI Model Routing Box Efficient Allocation.pdf")
        m = pc.FILENAME_RE.match(Path(name).stem)
        self.assertEqual(m["arxiv_id"], "2608.20316v1")

    def test_safe_filename_truncates(self):
        name = pa.safe_filename("2608.00001v2", "x" * 200)
        self.assertLessEqual(len(Path(name).stem), len("2608.00001v2_") + 80)

    def test_existing_papers_matches_any_version(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "2608.20316v1_Some Title.pdf").touch()
            Path(d, "notes.pdf").touch()
            found = pa.existing_papers(Path(d))
        self.assertEqual(list(found), ["2608.20316"])
        self.assertEqual(pa.base_id("2608.20316v3"), "2608.20316")


if __name__ == "__main__":
    unittest.main()
