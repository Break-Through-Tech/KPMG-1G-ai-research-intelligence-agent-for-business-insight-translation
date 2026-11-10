"""Regression coverage for passage provenance; these tests cannot prove entailment."""
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evidence import ABSTENTION, PARTIAL_NOTICE, render_evidence

# These historical refinement checks must not change the selected initial
# implementation. Load the preserved refined module explicitly for those tests.
_refined_path = (Path(__file__).resolve().parents[1] / "reports" / "summarization"
                 / "codex_v1_refined" / "implementation" / "evidence.py")
_refined_spec = importlib.util.spec_from_file_location("archived_refined_evidence", _refined_path)
archived_refined = importlib.util.module_from_spec(_refined_spec)
# Avoid writing a bytecode cache beside the preserved experiment files.
exec(compile(_refined_path.read_bytes(), str(_refined_path), "exec"), archived_refined.__dict__)


class EvidenceFixture:
    renderer = staticmethod(render_evidence)

    def setUp(self):
        self.hits = [
            SimpleNamespace(arxiv_id="2608.20318v1", chunk_id="a", text="The first method reduced latency by 30 percent in simulation."),
            SimpleNamespace(arxiv_id="2610.18843v1", chunk_id="b", text="The second method used calibrated cameras on a simulated rig."),
        ]

    def render(self, claims, coverage="full", max_chars=1500):
        return self.renderer(json.dumps({"coverage": coverage, "claims": claims}), self.hits, max_chars)

    def claim(self, source="S1", quote=None):
        return {"text": "The first method reduced simulation latency by 30 percent.",
                "evidence": [{"source": source, "quote": quote or self.hits[0].text}]}


class EvidenceTest(EvidenceFixture, unittest.TestCase):
    """Active codex_v1 initial provenance and rendering behavior."""

    def test_quote_from_wrong_source_does_not_leave_claim_or_citation(self):
        text, evidence, errors = self.render([self.claim(source="S2")])
        self.assertEqual((text, evidence), (ABSTENTION, []))
        self.assertIn("does not match", errors[0])

    def test_one_bad_evidence_link_rejects_whole_claim(self):
        claim = self.claim()
        claim["evidence"].append({"source": "S99", "quote": self.hits[0].text})
        text, evidence, errors = self.render([claim])
        self.assertEqual((text, evidence), (ABSTENTION, []))
        self.assertTrue(errors)

    def test_does_not_validate_against_hidden_truncated_text(self):
        text, evidence, errors = self.render([self.claim()], max_chars=25)
        self.assertEqual(text, ABSTENTION)
        self.assertTrue(errors)

    def test_whitespace_normalization_and_unique_citations(self):
        self.hits[0].text = self.hits[0].text.replace("method reduced", "method\n reduced")
        claim = self.claim(quote="The first method reduced latency by 30 percent in simulation.")
        claim["evidence"] *= 2
        text, evidence, errors = self.render([claim])
        self.assertEqual(text.count("[arXiv:2608.20318v1]"), 1)
        self.assertEqual(errors, [])

    def test_multi_source_claim_retains_source_associations(self):
        claim = {"text": "Two simulation methods were reported.", "evidence": [
            {"source": "S1", "quote": self.hits[0].text},
            {"source": "S2", "quote": self.hits[1].text},
        ]}
        text, evidence, errors = self.render([claim])
        self.assertIn("[arXiv:2608.20318v1] [arXiv:2610.18843v1]", text)
        self.assertEqual([x["chunk_id"] for x in evidence[0]["evidence"]], ["a", "b"])

    def test_malformed_outputs_abstain(self):
        for raw in ("", "not json", "[]", "null", '{"coverage":"full","claims":{}}',
                    '{"coverage":"full","claims":[]}', '{"coverage":"none","claims":[{}]}'):
            with self.subTest(raw=raw):
                text, evidence, errors = render_evidence(raw, self.hits, 1500)
                self.assertEqual((text, evidence), (ABSTENTION, []))
                self.assertTrue(errors)

    def test_explicit_abstention_and_fenced_json(self):
        raw = '```json\n{"coverage":"none","claims":[]}\n```'
        self.assertEqual(render_evidence(raw, self.hits, 1500), (ABSTENTION, [], []))

    def test_citations_cannot_bypass_source_mapping(self):
        for text in ("A finding [arXiv:2999.12345].", "A finding [1].", "See 2999.12345."):
            claim = self.claim()
            claim["text"] = text
            self.assertEqual(self.render([claim])[0], ABSTENTION)

    def test_keeps_independently_valid_claims_with_partial_notice(self):
        text, evidence, errors = self.render([self.claim(), self.claim(source="S99")])
        self.assertEqual(len(evidence), 1)
        self.assertEqual(len(errors), 1)
        self.assertTrue(text.endswith(PARTIAL_NOTICE))

    def test_short_or_edited_quote_is_rejected(self):
        for quote in ("latency", "The first method reduced latency by 50 percent in simulation."):
            self.assertEqual(self.render([self.claim(quote=quote)])[0], ABSTENTION)


class ArchivedRefinedEvidenceTest(EvidenceFixture, unittest.TestCase):
    """Refined-only behavior stays tested against its historical implementation."""

    renderer = staticmethod(archived_refined.render_evidence)

    def test_reference_heading_preserves_prose_but_rejects_reference_titles(self):
        own_result = self.hits[0].text
        title = "An agentic system for diagnosis with traceable reasoning."
        self.hits[0].text += " References [1] Zhao, W. " + title + " Nature 651 (2026)."
        self.assertEqual(archived_refined.research_text(self.hits[0], 1500), own_result)
        self.assertEqual(self.render([self.claim(quote=title)])[0], ABSTENTION)
        self.assertFalse(self.render([self.claim(quote=own_result)])[2])

    def test_heading_free_reference_list_cannot_support_claim(self):
        title = "Instance-aware rubric rewards for generated graphics."
        self.hits[0].text = ("[26] Hangyu Ran, Yuhao Zheng. " + title + " arXiv preprint, 2026. "
                             "[27] Zhihong Shao, Peiyi Wang. Mathematical reasoning. arXiv preprint, 2024.")
        self.assertEqual(archived_refined.research_text(self.hits[0], 1500), "")
        self.assertEqual(self.render([self.claim(quote=title)])[0], ABSTENTION)

    def test_normal_inline_citations_are_not_reference_list(self):
        text = "Our method improves on [1] and [2] in 2026, using two independent tests."
        self.hits[0].text = text
        self.assertEqual(archived_refined.research_text(self.hits[0], 1500), text)
        self.assertFalse(self.render([self.claim(quote=text)])[2])

    def test_initials_first_reference_list_is_not_research_evidence(self):
        self.hits[0].text = ("[1] K. Black, B. Liu. A robotics policy. arXiv preprint, 2026. "
                             "[2] M. J. Kim. Another robotics policy. Conference on Robotics, 2025.")
        self.assertEqual(archived_refined.research_text(self.hits[0], 1500), "")

    def test_source_quote_pair_recovery_still_checks_source_and_quote(self):
        claim = {"text": "A supported finding.", "evidence": ["S1", self.hits[0].text]}
        text, evidence, errors = self.render([claim])
        self.assertEqual(errors, [])
        self.assertIn("[arXiv:2608.20318v1]", text)
        self.assertEqual(evidence[0]["normalization"], "source_quote_pair")
        claim["evidence"][0] = "S2"
        self.assertEqual(self.render([claim])[0], ABSTENTION)
        claim["evidence"] = ["S1", "A made-up quote that was never supplied."]
        self.assertEqual(self.render([claim])[0], ABSTENTION)


if __name__ == "__main__":
    unittest.main()
