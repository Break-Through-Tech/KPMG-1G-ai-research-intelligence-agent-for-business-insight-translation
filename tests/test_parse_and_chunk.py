"""Run from the repo root: python -m unittest discover -s tests -v"""

import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import parse_and_chunk as pc


def body(n: int) -> list[str]:
    return [f"Paragraph line {chr(65 + i)} about topic {chr(97 + n)}." for i in range(6)]


def page(n: int, header: str = "My Paper Title: A Study") -> str:
    return "\n".join([header, *body(n), str(n)])


class StripRunningLinesTest(unittest.TestCase):
    def test_removes_repeated_header_and_page_numbers(self):
        cleaned = pc.strip_running_lines([page(i) for i in range(1, 9)])
        for i, lines in enumerate(cleaned, start=1):
            self.assertEqual(lines, body(i))

    def test_removes_arxiv_stamp(self):
        pages = ["Intro text\narXiv:2608.20316v1  [cs.AI]  20 Aug 2026"] + ["filler"] * 3
        self.assertEqual(pc.strip_running_lines(pages)[0], ["Intro text"])

    def test_keeps_lines_that_repeat_mid_page(self):
        pages = [page(i).replace("Paragraph line C", "Table 1") for i in range(1, 9)]
        self.assertTrue(all(any(ln.startswith("Table 1") for ln in lines) for lines in pc.strip_running_lines(pages)))

    def test_keeps_short_pages_intact(self):
        self.assertEqual(pc.strip_running_lines(["Title", "Title"] * 2)[0], ["Title"])


class BackMatterTest(unittest.TestCase):
    def make(self, tail):
        body = [[f"Body line {i}" for i in range(20)]]
        return body + [tail]

    def test_cuts_references_and_keeps_appendix(self):
        pages = self.make(
            ["Conclusion text.", "References", "[1] Smith. 2024.", "A Proof of Theorem 1", "Appendix text."]
        )
        lines, refs, appendix = pc.remove_back_matter(pages)
        text = [ln for _, _, ln in lines]
        self.assertTrue(refs and appendix)
        self.assertNotIn("[1] Smith. 2024.", text)
        self.assertIn("Conclusion text.", text)
        self.assertEqual([s for _, s, ln in lines if ln == "Appendix text."], ["appendix"])

    def test_cuts_acknowledgments_before_references(self):
        pages = self.make(["Last result.", "Acknowledgments", "Funded by a grant.", "References", "[1] Doe."])
        text = [ln for _, _, ln in pc.remove_back_matter(pages)[0]]
        self.assertIn("Last result.", text)
        self.assertNotIn("Funded by a grant.", text)

    def test_no_references_heading_keeps_everything(self):
        pages = self.make(["Just text."])
        lines, refs, _ = pc.remove_back_matter(pages)
        self.assertFalse(refs)
        self.assertEqual(len(lines), 21)


class TextTest(unittest.TestCase):
    def test_join_lines_repairs_hyphenation(self):
        self.assertEqual(pc.join_lines(["reinforce-", "ment learning", "is fun"]), "reinforcement learning is fun")

    def test_join_lines_keeps_real_hyphens(self):
        self.assertEqual(pc.join_lines(["GPT-", "4o"]), "GPT- 4o")  # next line not lowercase, leave it

    def test_split_sentences_tracks_pages(self):
        lines = [(1, "First sentence. Second"), (1, "sentence ends here."), (2, "Third one.")]
        self.assertEqual(
            pc.split_sentences(lines),
            [(1, "First sentence."), (1, "Second sentence ends here."), (2, "Third one.")],
        )


class PackChunksTest(unittest.TestCase):
    def setUp(self):
        self.sentences = [(i // 5 + 1, " ".join(f"s{i}w{j}" for j in range(10)) + ".") for i in range(30)]

    def test_respects_max_words(self):
        chunks = pc.pack_chunks(self.sentences, max_words=40, overlap_words=10)
        self.assertTrue(all(len(text.split()) <= 40 for _, _, text in chunks))

    def test_overlap_repeats_last_sentence(self):
        chunks = pc.pack_chunks(self.sentences, max_words=40, overlap_words=10)
        first_tail = chunks[0][2].split()[-10:]
        self.assertEqual(chunks[1][2].split()[:10], first_tail)

    def test_covers_every_sentence(self):
        chunks = pc.pack_chunks(self.sentences, max_words=40, overlap_words=10)
        joined = " ".join(text for _, _, text in chunks)
        for _, sent in self.sentences:
            self.assertIn(sent, joined)

    def test_hard_splits_long_sentence(self):
        long = [(1, " ".join(["word"] * 250))]
        chunks = pc.pack_chunks(long, max_words=100, overlap_words=0)
        self.assertEqual([len(t.split()) for _, _, t in chunks], [100, 100, 50])

    def test_page_range(self):
        start, end, _ = pc.pack_chunks(self.sentences, max_words=80, overlap_words=0)[0]
        self.assertEqual((start, end), (1, 2))


class FilterTest(unittest.TestCase):
    def test_garbled_detection(self):
        self.assertTrue(pc.is_garbled("wepropose G-CARL,agroundedchecklistalignedreinforcementlearningframework"))
        self.assertFalse(pc.is_garbled("We propose a grounded, checklist-aligned reinforcement learning framework."))

    def test_near_duplicate(self):
        base = " ".join(f"token{i}" for i in range(120))
        dedup = pc.NearDuplicateFilter(threshold=0.85)
        self.assertFalse(dedup.is_duplicate(base))
        self.assertTrue(dedup.is_duplicate(base.replace("token60", "changed")))  # one word differs
        self.assertTrue(dedup.is_duplicate(base.upper()))  # exact after normalizing
        self.assertFalse(dedup.is_duplicate(" ".join(f"other{i}" for i in range(120))))


@unittest.skipUnless(list((REPO / "data").glob("*.pdf")), "no PDFs in data/")
class CorpusTest(unittest.TestCase):
    def test_runs_on_seeded_papers(self):
        with tempfile.TemporaryDirectory() as tmp:
            reports = pc.run(REPO / "data", Path(tmp), max_words=120, overlap_words=20, dedup_threshold=0.85)
            chunks = (Path(tmp) / "chunks.jsonl").read_text().splitlines()
        self.assertEqual(len(reports), len(list((REPO / "data").glob("*.pdf"))))
        self.assertTrue(all(r.chunks > 0 for r in reports))
        self.assertTrue(all(r.references_removed for r in reports))
        self.assertEqual(len(chunks), sum(r.chunks for r in reports))


if __name__ == "__main__":
    unittest.main()
