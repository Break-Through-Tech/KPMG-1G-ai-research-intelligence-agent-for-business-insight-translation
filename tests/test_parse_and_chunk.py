"""Run from the repo root: python -m unittest discover -s tests -v"""

import json
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

    def test_header_threshold_follows_min_share(self):
        # 2 of 5 pages is 40%, above the 30% default, so the header goes.
        pages = [page(i, header="Short Header") if i <= 2 else page(i, header=f"Unique {i}") for i in range(1, 6)]
        self.assertNotIn("Short Header", pc.strip_running_lines(pages)[0])
        # 3 of 11 pages is 27%, below 30%, so it stays.
        pages = [page(i, header="Short Header") if i <= 3 else page(i, header=f"Unique {i}") for i in range(1, 12)]
        self.assertIn("Short Header", pc.strip_running_lines(pages)[0])

    def test_keeps_short_pages_intact(self):
        self.assertEqual(pc.strip_running_lines(["Title", "Title"] * 2)[0], ["Title"])


class BackMatterTest(unittest.TestCase):
    def make(self, tail):
        body = [[f"Body line {i}" for i in range(20)]]
        return body + [tail]

    def test_cuts_references_and_keeps_appendix(self):
        pages = self.make(
            ["Conclusion text.", "References", "[1] Smith. 2024.", "[2] Lee. 2023.", "A Proof of Theorem 1", "Appendix text."]
        )
        lines, refs, appendix = pc.remove_back_matter(pages)
        text = [ln for _, _, ln in lines]
        self.assertTrue(refs and appendix)
        self.assertNotIn("[1] Smith. 2024.", text)
        self.assertIn("Conclusion text.", text)
        self.assertEqual([s for _, s, ln in lines if ln == "Appendix text."], ["appendix"])

    def test_cuts_acknowledgments_before_references(self):
        pages = self.make(["Last result.", "Acknowledgments", "Funded by a grant.", "References", "[1] Doe. 2022.", "[2] Roe. 2024."])
        text = [ln for _, _, ln in pc.remove_back_matter(pages)[0]]
        self.assertIn("Last result.", text)
        self.assertNotIn("Funded by a grant.", text)

    def test_citation_after_references_is_not_an_appendix(self):
        pages = self.make(["Conclusion text.", "References", "A. Smith and Jones", "Some venue, 2024.", "B. Lee. Other venue, 2023."])
        lines, refs, appendix = pc.remove_back_matter(pages)
        self.assertTrue(refs)
        self.assertFalse(appendix)
        self.assertNotIn("A. Smith and Jones", [ln for _, _, ln in lines])

    def test_appendix_heading_forms(self):
        for line in ["Appendix B: Extra results", "A Proof of Theorem 1", "A. Implementation Details", "A Additional Results 19"]:
            self.assertTrue(pc.is_appendix_heading(line), line)
        for line in ["A. Smith and Jones", "A. Vaswani, N. Shazeer", "A Smith et al"]:
            self.assertFalse(pc.is_appendix_heading(line), line)

    def test_toc_entry_without_citations_is_not_the_reference_list(self):
        toc = [["Title", "Abstract text."], ["Contents", "1 Introduction", "References", "A Proofs"]]
        body = [[f"Body line {i}" for i in range(10)]]
        refs = [["References", "[1] Smith. 2024.", "[2] Lee. 2023."]]
        lines, found, _ = pc.remove_back_matter(toc + body + refs)
        text = [ln for _, _, ln in lines]
        self.assertTrue(found)
        self.assertIn("Body line 9", text)
        self.assertNotIn("[1] Smith. 2024.", text)

    def test_references_early_in_a_paper_with_a_long_appendix_are_found(self):
        # 2610.35715v1: references end on page 10 of 69, at 8% of the lines.
        pages = [["Title"], ["Body."], ["References", "[1] Smith. 2024.", "[2] Lee. 2023."], ["A Proofs"]]
        pages += [[f"Appendix line {i}" for i in range(40)] for _ in range(10)]
        lines, found, appendix = pc.remove_back_matter(pages)
        self.assertTrue(found and appendix)
        self.assertNotIn("[1] Smith. 2024.", [ln for _, _, ln in lines])

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


SEEDED = ("2608.20316v1", "2608.20318v1", "2608.20319v1", "2608.20320v1", "2608.20331v1")


@unittest.skipUnless(list((REPO / "data").glob("*.pdf")), "no PDFs in data/")
class CorpusTest(unittest.TestCase):
    """Runs on the 5 seeded papers that are committed to the repo, not on every PDF in data/,
    so it stays fast and gives the same result after a 50-paper pull."""

    @classmethod
    def setUpClass(cls):
        import shutil

        cls.tmp = tempfile.TemporaryDirectory()
        src = Path(cls.tmp.name) / "in"
        src.mkdir()
        for arxiv_id in SEEDED:
            for pdf in (REPO / "data").glob(f"{arxiv_id}_*.pdf"):
                shutil.copy(pdf, src / pdf.name)
        cls.n_pdfs = len(list(src.glob("*.pdf")))
        cls.reports = pc.run(src, Path(cls.tmp.name) / "out", max_words=120, overlap_words=20, dedup_threshold=0.85)
        cls.chunks = (Path(cls.tmp.name) / "out" / "chunks.jsonl").read_text().splitlines()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_runs_on_seeded_papers(self):
        self.assertEqual(self.n_pdfs, len(SEEDED))
        self.assertEqual(len(self.reports), len(SEEDED))
        self.assertTrue(all(r.chunks > 0 for r in self.reports))
        # The 5 seeded papers all have a References heading.
        self.assertTrue(all(r.references_removed for r in self.reports))
        self.assertEqual(len(self.chunks), sum(r.chunks for r in self.reports))

    def _text(self, arxiv_id: str) -> str:
        return " ".join(json.loads(c)["text"] for c in self.chunks if json.loads(c)["arxiv_id"] == arxiv_id)

    def _needs_pdfium(self):
        try:
            import pypdfium2  # noqa: F401
        except ImportError:
            self.skipTest("pypdfium2 not installed")

    def test_gcarl_parses_cleanly(self):
        # pypdf drops the word spacing on 5 G-CARL pages. With pypdf only 25 of its chunks
        # survived; with pypdf + page fallback 49; with pypdfium2 as the primary extractor
        # no chunk is lost to garbling and the abstract is kept.
        self._needs_pdfium()
        gcarl = next(r for r in self.reports if r.arxiv_id == "2608.20331v1")
        self.assertEqual(gcarl.extractor, "pypdfium2")
        self.assertEqual(gcarl.status, "ok")
        self.assertEqual(gcarl.dropped_garbled, 0)
        self.assertGreater(gcarl.chunks, 50)
        self.assertIn("Addressing this need requires both evidence-grounded medical factuality", self._text("2608.20331v1"))

    def test_words_pypdf_joined_on_normal_pages_are_separate(self):
        # pypdf output for AI4AI-Bench p3 and p12, which is_garbled() does not flag.
        self._needs_pdfium()
        text = self._text("2608.20318v1")
        for joined in ("agentsdo", "Evenautoresearch", "win.Every"):
            self.assertNotIn(joined, text)
        self.assertIn("Even autoresearch", text)
        self.assertIn("what agents do", text)

    def test_line_hyphens_split_words_and_keep_compounds(self):
        # pdfium writes words split across lines as "learn\ufffeing" on one line. Real output,
        # not a hand-built string: these come from G-CARL and Pandora's Box.
        self._needs_pdfium()
        gcarl = self._text("2608.20331v1")
        self.assertNotIn("\ufffe", gcarl)
        for split in ("learn-ing", "fac-tuality", "ex-isting", "cap-ture", "intro-duce"):
            self.assertNotIn(split, gcarl)
        self.assertIn("factuality", gcarl)
        self.assertIn("a real-world PMRI benchmark", gcarl)
        self.assertIn("a retrieval-augmented specialist", self._text("2608.20316v1"))


class ExtractionFallbackTest(unittest.TestCase):
    """pypdfium2 is primary; pypdf is used for the whole file without pypdfium2, and for
    single pages where pypdfium2 returns nothing usable."""

    GOOD = "This page came out of pdfium with normal spacing between all of its words."
    GLUED = "Thispagecameoutofpdfiumwithnospacingbetweenanyofitswordsatall andsomemoreglued textherewithoutspaces"
    PYPDF = "This page came out of pypdf with the spacing intact between all of its words."

    def _run(self, pypdf_pages, pdfium_pages):
        class FakePage:
            def __init__(self, text):
                self.text = text

            def extract_text(self):
                return self.text

        class FakeReader:
            def __init__(self, _path):
                self.pages = [FakePage(t) for t in pypdf_pages]

        orig_reader, orig_pdfium = pc.PdfReader, pc._pdfium_pages
        pc.PdfReader, pc._pdfium_pages = FakeReader, lambda _path: pdfium_pages
        try:
            return pc.extract_pages(Path("fake.pdf"))
        finally:
            pc.PdfReader, pc._pdfium_pages = orig_reader, orig_pdfium

    def test_pdfium_is_used_when_its_pages_are_fine(self):
        pages, replaced, extractor = self._run([self.PYPDF, self.PYPDF], [self.GOOD, self.GOOD])
        self.assertEqual((pages, replaced, extractor), ([self.GOOD, self.GOOD], 0, "pypdfium2"))

    def test_only_the_weak_pdfium_page_is_replaced(self):
        pages, replaced, _ = self._run([self.PYPDF, self.PYPDF], [self.GOOD, self.GLUED])
        self.assertEqual((pages, replaced), ([self.GOOD, self.PYPDF], 1))
        pages, replaced, _ = self._run([self.PYPDF, self.PYPDF], [self.GOOD, "  \n"])
        self.assertEqual((pages, replaced), ([self.GOOD, self.PYPDF], 1))

    def test_pypdf_page_that_is_also_garbled_is_not_used(self):
        self.assertEqual(self._run([self.GLUED], [self.GLUED]), ([self.GLUED], 0, "pypdfium2"))

    def test_missing_pdfium_uses_pypdf_for_the_whole_file(self):
        self.assertEqual(self._run([self.PYPDF, self.GLUED], None), ([self.PYPDF, self.GLUED], 0, "pypdf"))

    def test_page_count_mismatch_keeps_pdfium(self):
        self.assertEqual(self._run([self.PYPDF], [self.GOOD, self.GLUED]), ([self.GOOD, self.GLUED], 0, "pypdfium2"))


class LineHyphenTest(unittest.TestCase):
    """pdfium marks a hyphen at a line break as U+FFFE; the paper's own words decide."""

    def resolve(self, text: str, context: str = "") -> str:
        return pc.resolve_line_hyphens([text, context])[0]

    def test_split_word_is_joined(self):
        self.assertEqual(self.resolve("we use fac\ufffetuality here"), "we use factuality here")

    def test_joined_form_seen_elsewhere_is_joined(self):
        self.assertEqual(self.resolve("deep learn\ufffeing", "learning is used"), "deep learning")

    def test_hyphenated_form_seen_elsewhere_keeps_hyphen(self):
        self.assertEqual(self.resolve("in real\ufffeworld use", "real-world data"), "in real-world use")

    def test_both_halves_seen_as_words_keep_hyphen(self):
        self.assertEqual(self.resolve("for decision\ufffemakers", "a decision by the makers"), "for decision-makers")

    def test_capital_second_half_keeps_hyphen(self):
        self.assertEqual(self.resolve("Megatron\ufffeLM partitions"), "Megatron-LM partitions")

    def test_mark_before_a_digit_is_dropped(self):
        # A footnote number interrupted "trav-eler"; no rule applies, the mark just goes.
        self.assertEqual(self.resolve("trav\ufffe2 eler"), "trav2 eler")



if __name__ == "__main__":
    unittest.main()


class CliValidationTest(unittest.TestCase):
    def test_rejects_bad_numbers(self):
        for args in (["--max-words", "0"], ["--max-words", "-5"], ["--overlap-words", "120"], ["--dedup-threshold", "0"]):
            with self.assertRaises(SystemExit), open("/dev/null", "w") as devnull:
                old, sys.stderr = sys.stderr, devnull
                try:
                    pc.main(args)
                finally:
                    sys.stderr = old

