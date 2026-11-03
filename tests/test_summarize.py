"""Summarization tests. No API key, model or database needed: python -m unittest discover -s tests -v"""

import contextlib
import io
import json
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import summarize as sm


@dataclass
class FakeHit:
    arxiv_id: str
    title: str = "A paper"
    page_start: int = 1
    page_end: int = 2
    text: str = "Some chunk text."
    similarity: float = 0.6
    rank: int = 1
    score: float = 0.6


@dataclass
class FakeMessage:
    text: str


class FakeLLM:
    """Returns a fixed reply and remembers what it was sent."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return FakeMessage(self.reply)


def search_returning(hits):
    def search(question, k, max_per_paper=None):
        search.calls.append({"k": k, "max_per_paper": max_per_paper})
        return hits[:k]

    search.calls = []
    return search


class CitationTest(unittest.TestCase):
    def test_extracts_ids_in_order_without_duplicates(self):
        text = "A [arXiv:2608.20318v1]. B [arXiv:2608.20316v1] [arXiv:2608.20318v1]. C [arxiv: 2610.18842]."
        self.assertEqual(sm.extract_citations(text), ["2608.20318v1", "2608.20316v1", "2610.18842"])

    def test_reads_several_ids_in_one_bracket(self):
        text = "A [arXiv:2608.20318v1, arXiv:2608.20316v1]. B [arXiv: 2610.18842v1; 2610.18843v1]."
        self.assertEqual(
            sm.extract_citations(text), ["2608.20318v1", "2608.20316v1", "2610.18842v1", "2610.18843v1"]
        )

    def test_reads_ids_without_the_arxiv_prefix(self):
        self.assertEqual(sm.extract_citations("A [2608.20318v1]."), ["2608.20318v1"])

    def test_ignores_ids_and_numbers_outside_citations(self):
        self.assertEqual(sm.extract_citations("see 2608.20318, source [1], score [0.166]"), [])


class PromptTest(unittest.TestCase):
    def test_sources_carry_id_title_and_pages(self):
        hits = [FakeHit("2608.20318v1", "AI4AI-Bench", 3, 4, "Mean score 0.166.")]
        out = sm.format_sources(hits)
        self.assertIn("[1] ID: 2608.20318v1 | AI4AI-Bench | pages 3-4", out)
        self.assertIn("Mean score 0.166.", out)

    def test_long_chunks_are_cut(self):
        out = sm.format_sources([FakeHit("x", text="word " * 1000)], max_chars=100)
        self.assertLess(len(out), 200)
        self.assertTrue(out.endswith("..."))

    def test_question_goes_in_the_human_message(self):
        messages = sm.build_messages("Why?", [FakeHit("x")])
        self.assertEqual([role for role, _ in messages], ["system", "human"])
        self.assertTrue(messages[1][1].endswith("Question: Why?"))


class SummarizeTest(unittest.TestCase):
    def test_flags_uncited_and_made_up_papers(self):
        hits = [FakeHit("2608.20318v1"), FakeHit("2608.20318v1"), FakeHit("2608.20316v1")]
        llm = FakeLLM("Finding one [arXiv:2608.20318]. Finding two [2999.00001v1].")
        answer = sm.summarize("q", search_returning(hits), llm, k=3, max_per_paper=None)

        self.assertEqual(answer.retrieved_ids, ["2608.20318v1", "2608.20316v1"])
        self.assertEqual(answer.cited_ids, ["2608.20318", "2999.00001v1"])
        self.assertEqual(answer.unknown_citations, ["2999.00001v1"])
        self.assertEqual(answer.uncited_sources, ["2608.20316v1"])
        self.assertEqual(len(llm.calls), 1)

    def test_no_hits_skips_the_model(self):
        llm = FakeLLM("should not be called")
        answer = sm.summarize("q", search_returning([]), llm)
        self.assertEqual(llm.calls, [])
        self.assertIn("nothing to summarize", answer.text)


class PromptV2Test(unittest.TestCase):
    def test_v2_is_the_default_and_v1_is_kept(self):
        self.assertIn("Business implications", sm.build_messages("q", [FakeHit("x")])[0][1])
        self.assertEqual(sm.build_messages("q", [FakeHit("x")], prompt="v1")[0][1], sm.PROMPT_V1)

    def test_v2_covers_the_advisor_answer_pattern_and_stress_tests(self):
        for part in ("**Answer:**", "**Evidence:**", "**Limitations:**", "**Business implications:**",
                     sm.NOT_COVERED, "partly", "assumes", "do not support", "ambiguous"):
            self.assertIn(part, sm.PROMPT_V2)

    def test_sources_carry_author_and_date_when_known(self):
        info = {"2608.20318v1": {"first_author": "Yizhe Chi et al.", "published": "2026-08-20"}}
        out = sm.format_sources([FakeHit("2608.20318v1", "AI4AI-Bench", 3, 4)], paper_info=info)
        self.assertIn("[1] ID: 2608.20318v1 | AI4AI-Bench | Yizhe Chi et al. | 2026-08-20 | pages 3-4", out)

    def test_load_paper_info_reads_first_author(self):
        import tempfile, os
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("arxiv_id,authors,published\n2608.20318v1,A One; B Two,2026-08-20\n2608.20316v1,Solo,2026-08-21\n")
        try:
            info = sm.load_paper_info(Path(f.name))
        finally:
            os.unlink(f.name)
        self.assertEqual(info["2608.20318v1"]["first_author"], "A One et al.")
        self.assertEqual(info["2608.20316v1"]["first_author"], "Solo")
        self.assertEqual(sm.load_paper_info(Path("no/such/file.csv")), {})


class AbstentionTest(unittest.TestCase):
    def test_relevance_floor_stops_before_the_model(self):
        llm = FakeLLM("should not be called")
        hits = [FakeHit("2608.20318v1", similarity=0.30), FakeHit("2608.20316v1", similarity=0.25)]
        answer = sm.summarize("quantum?", search_returning(hits), llm, relevance_floor=0.38)
        self.assertEqual(llm.calls, [])
        self.assertTrue(answer.abstained and answer.stopped_at_retrieval)
        self.assertTrue(answer.text.startswith(sm.NOT_COVERED))
        self.assertIn("2608.20318v1", answer.text)

    def test_above_the_floor_goes_to_the_model(self):
        llm = FakeLLM("**Answer:** yes [arXiv:2608.20318v1].")
        hits = [FakeHit("2608.20318v1", similarity=0.30), FakeHit("2608.20316v1", similarity=0.50)]
        answer = sm.summarize("q", search_returning(hits), llm, relevance_floor=0.38)
        self.assertEqual(len(llm.calls), 1)
        self.assertFalse(answer.abstained)

    def test_model_abstention_is_detected_for_both_prompts(self):
        declines = [
            sm.NOT_COVERED + "\nThe closest topic they cover is robot learning.",
            sm.NOT_COVERED + " The closest topic is robotics.",
            "**The retrieved papers do not cover this question.**",
            "**Answer:** The retrieved papers do not cover this question.",
            "Answer: The retrieved papers do not cover this question.",
            "## The retrieved papers do not cover this question.",
            '"The retrieved papers do not cover this question."',
            "\n\n  The retrieved papers do not cover this question.",
            "The retrieved papers do not answer this.",
        ]
        for text in declines:
            self.assertTrue(sm.is_abstention(text), text)

    def test_partial_coverage_and_normal_answers_are_not_declines(self):
        answers = [
            "The retrieved papers do not cover this question directly, but two of them study routing [arXiv:2608.20316v1].",
            "**Answer:** The retrieved papers show a 31% cut [arXiv:2610.18862v1].",
            "**Answer:** Yes.\n\nThe retrieved papers do not cover this question. (quoted later, not first)",
            "",
        ]
        for text in answers:
            self.assertFalse(sm.is_abstention(text), text)

    def test_search_gets_k_and_the_per_paper_cap(self):
        search = search_returning([FakeHit("x")])
        sm.summarize("q", search, FakeLLM("**Answer:** ok [arXiv:x]."))
        self.assertEqual(search.calls, [{"k": sm.DEFAULT_K, "max_per_paper": sm.DEFAULT_MAX_PER_PAPER}])


class BatchTest(unittest.TestCase):
    def test_batch_writes_one_scored_row_per_question(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            bench = Path(tmp) / "bench.jsonl"
            bench.write_text(
                json.dumps({"id": "q01", "question": "a?", "relevant_arxiv_ids": ["2608.20318v1"], "expected_answer": "x"}) + "\n"
                + json.dumps({"id": "q02", "question": "b?", "relevant_arxiv_ids": ["2608.20316v1"]}) + "\n"
            )

            def answer_one(q):
                return sm.Answer(question=q, text="t", cited_ids=["2608.20318"], retrieved_ids=["2608.20318v1"])

            with contextlib.redirect_stdout(io.StringIO()):
                out = sm.run_batch(bench, answer_one, Path(tmp) / "out", "v2", pause=0)
            rows = [json.loads(line) for line in out.read_text().splitlines()]
        self.assertEqual([r["id"] for r in rows], ["q01", "q02"])
        self.assertEqual(rows[0]["relevant_cited"], ["2608.20318"])
        self.assertEqual(rows[1]["relevant_cited"], [])
        self.assertIsNone(rows[0]["rubric_accuracy"])
        self.assertEqual(rows[0]["prompt"], "v2")
        self.assertEqual(rows[0]["error"], "")

    def test_batch_keeps_going_after_an_api_error(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            bench = Path(tmp) / "bench.jsonl"
            bench.write_text("".join(json.dumps({"id": f"q0{i}", "question": f"{i}?"}) + "\n" for i in (1, 2)))

            def answer_one(q):
                if q == "1?":
                    raise RuntimeError("429 quota exceeded")
                return sm.Answer(question=q, text="ok")

            with contextlib.redirect_stdout(io.StringIO()):
                out = sm.run_batch(bench, answer_one, Path(tmp) / "out", "v2", pause=0)
            rows = [json.loads(line) for line in out.read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertIn("429", rows[0]["error"])
        self.assertEqual(rows[1]["answer"], "ok")


if __name__ == "__main__":
    unittest.main()
