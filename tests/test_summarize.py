"""Summarization tests. No API key, model or database needed: python -m unittest discover -s tests -v"""

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
    def search(question, k):
        return hits[:k]

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

    def test_malformed_ids_are_not_truncated_into_valid_citations(self):
        for token in ("2608.203160", "12608.20318", "2608.20318v1extra", "2608.20318v",
                      "２６０８.２０３１８", "2608.20318.5"):
            with self.subTest(token=token):
                self.assertEqual(sm.extract_citations(f"Finding [arXiv:{token}]."), [])


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
        answer = sm.summarize("q", search_returning(hits), llm, k=3)

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

    def test_codex_renders_citations_from_passages_in_one_call(self):
        import json
        hit = FakeHit("2608.20318v1", text="In simulation, success fell to 0.509 under severe rig perturbations.")
        reply = {"coverage": "partial", "claims": [{
            "text": "In simulation, severe perturbations reduced success to 0.509.",
            "evidence": [{"source": "S1", "quote": hit.text}],
        }]}
        llm = FakeLLM(json.dumps(reply))
        answer = sm.summarize("How reliable?", search_returning([hit]), llm, 10, prompt_version="codex_v1")
        self.assertEqual(answer.cited_ids, [hit.arxiv_id])
        self.assertEqual(answer.validation_errors, [])
        self.assertEqual(answer.evidence[0]["evidence"][0]["arxiv_id"], hit.arxiv_id)
        self.assertIn("only part of the question", answer.text)
        self.assertEqual(len(llm.calls), 1)
        self.assertIn("Passage S1 | arXiv ID:", llm.calls[0][1][1])

    def test_unknown_prompt_rejected_before_search_or_model_call(self):
        with self.assertRaises(ValueError):
            sm.summarize("q", lambda *_: self.fail("search called"), FakeLLM(""), prompt_version="typo")

    def test_supports_text_methods_and_content_blocks(self):
        class MethodMessage:
            def text(self):
                return " method text "
        class BlocksMessage:
            content = [{"type": "reasoning", "text": "internal"}, {"type": "text", "text": "answer"}]
        self.assertEqual(sm.response_text(MethodMessage()), "method text")
        self.assertEqual(sm.response_text(BlocksMessage()), "answer")
        self.assertEqual(sm.response_text(object()), "")

    def test_callable_string_text_is_read_as_property(self):
        class CallableText(str):
            def __call__(self):
                raise AssertionError("Deprecated callable text was invoked")
        class Message:
            text = CallableText(" reply ")
        self.assertEqual(sm.response_text(Message()), "reply")


if __name__ == "__main__":
    unittest.main()
