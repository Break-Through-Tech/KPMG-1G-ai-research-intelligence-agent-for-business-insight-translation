"""Pure unit tests for benchmark citation checks; no Google API calls."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate_summaries import (
    PROMPTS, SYSTEM_PROMPT, TUNED_PROMPT, TUNED_V2_PROMPT, TUNED_V3_PROMPT,
    PromptOverrideLLM, RecordingLLM, corpus_coverage, load_previous, metrics,
    preserve_run_manifest, validate_previous,
)


class CitationMetricsTest(unittest.TestCase):
    def test_single_paper_correct(self):
        m = metrics(["2608.20318v1"], ["2608.20318v1"], ["2608.20318v1"])
        self.assertEqual(m["citation_precision"], 1)
        self.assertEqual(m["citation_recall"], 1)
        self.assertTrue(m["all_relevant_cited"])

    def test_multi_paper_partial(self):
        m = metrics(["2608.20318v1", "2610.18842v1"], ["2608.20318v1"], ["2608.20318v1"])
        self.assertEqual(m["retrieval_recall"], 0.5)
        self.assertEqual(m["citation_recall"], 0.5)
        self.assertFalse(m["all_relevant_cited"])

    def test_off_target_vs_unretrieved(self):
        m = metrics(["2608.20318v1"], ["2608.20318v1", "2610.18842v1"],
                    ["2608.20318v1", "2610.18842v1", "2608.20331v1"])
        self.assertEqual(m["citation_precision"], 0.3333)
        self.assertEqual(m["off_target_citation_ids"], ["2608.20331", "2610.18842"])
        self.assertEqual(m["unretrieved_citation_ids"], ["2608.20331"])

    def test_uncited_answer(self):
        m = metrics(["2608.20318v1"], ["2608.20318v1"], [])
        self.assertIsNone(m["citation_precision"])
        self.assertEqual(m["citation_recall"], 0)
        self.assertTrue(m["no_citations"])

    def test_version_normalization(self):
        m = metrics(["2608.20318v1"], ["2608.20318v2"], ["2608.20318v2"])
        self.assertTrue(m["all_relevant_retrieved"])
        self.assertTrue(m["all_relevant_cited"])

    def test_missing_index_papers(self):
        questions = [{"relevant_arxiv_ids": ["2608.20318v1", "2610.18842v1"]}]
        ids, missing = corpus_coverage(questions, {"2608.20318"})
        self.assertEqual(len(ids), 2)
        self.assertEqual(missing, ["2610.18842"])

    def test_prompt_override(self):
        class Dummy:
            def invoke(self, messages):
                return messages
        out = PromptOverrideLLM(Dummy(), "new system").invoke(
            [("system", "old"), ("human", "question")])
        self.assertEqual(out, [("system", "new system"), ("human", "question")])

    def test_tuned_v2_is_registered(self):
        self.assertIn("tuned_v2", PROMPTS)
        self.assertEqual(PROMPTS["tuned_v2"], TUNED_V2_PROMPT)

    def test_tuned_v3_is_registered(self):
        self.assertIn("tuned_v3", PROMPTS)
        self.assertEqual(PROMPTS["tuned_v3"], TUNED_V3_PROMPT)
        self.assertIn("bibliography", TUNED_V3_PROMPT)
        self.assertIn("directly supports the specific claim", TUNED_V3_PROMPT)
        self.assertIn("If only part of the question is answered", TUNED_V3_PROMPT)

    def test_codex_v1_is_registered(self):
        from evidence import CODEX_V1_PROMPT
        self.assertEqual(PROMPTS["codex_v1"], CODEX_V1_PROMPT)

    def test_v2_clears_only_numeric_source_headers(self):
        class Dummy:
            def invoke(self, messages):
                return messages

        messages = [
            ("system", "old"),
            ("human", "Sources:\n\n[1] ID: 2608.20316v1 | Paper | pages 1-2\n"
                      "The paper cites [1] and [6] in its bibliography.\n\n"
                      "[2] ID: 2608.20318v1 | Another paper | pages 3-4\n"
                      "This finding cites [1, 10].\n\nQuestion: What happened?"),
        ]
        out = PromptOverrideLLM(Dummy(), "v2", clear_source_numbers=True).invoke(messages)
        self.assertEqual(
            out[1][1],
            "Sources:\n\nSource 1 | arXiv ID: 2608.20316v1 | Paper | pages 1-2\n"
            "The paper cites [1] and [6] in its bibliography.\n\n"
            "Source 2 | arXiv ID: 2608.20318v1 | Another paper | pages 3-4\n"
            "This finding cites [1, 10].\n\nQuestion: What happened?",
        )

    def test_baseline_and_tuned_prompts_and_headers_stay_unchanged(self):
        class Dummy:
            def invoke(self, messages):
                return messages

        source_message = "Sources:\n\n[1] ID: 2608.20316v1 | Paper | pages 1-2\nText."
        for version, prompt in (("baseline", SYSTEM_PROMPT), ("tuned", TUNED_PROMPT)):
            self.assertEqual(PROMPTS[version], prompt)
            out = PromptOverrideLLM(Dummy(), prompt).invoke(
                [("system", "old"), ("human", source_message)])
            self.assertEqual(out, [("system", prompt), ("human", source_message)])


if __name__ == "__main__":
    unittest.main()

# Test export file preservation separately from runtime API integration.
class ReviewCsvTest(unittest.TestCase):
    def test_review_resume_adds_questions_but_preserves_scores(self):
        import csv
        import tempfile
        from evaluate_summaries import write_review_csv
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "review.csv"
            base = {"id": "q01", "question": "Q1", "expected_answer": "A1", "answer": "one",
                    "relevant_arxiv_ids": [], "retrieved_arxiv_ids": [], "cited_arxiv_ids": [],
                    "correct_citation_ids": [], "off_target_citation_ids": [], "unretrieved_citation_ids": [],
                    "citation_precision": None}
            write_review_csv(path, [base])
            with path.open(newline="") as f:
                rows = list(csv.DictReader(f))
            rows[0]["accuracy_1_to_5"] = "4"
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            newer = dict(base, id="q02", question="Q2")
            write_review_csv(path, [base, newer])
            with path.open(newline="") as f:
                result = list(csv.DictReader(f))
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["accuracy_1_to_5"], "4")


class ResumeSafetyTest(unittest.TestCase):
    def setUp(self):
        import hashlib
        self.benchmark = [
            {"id": "q01", "question": "First?", "relevant_arxiv_ids": ["2608.20316v1"], "expected_answer": "One."},
            {"id": "q02", "question": "Second?", "relevant_arxiv_ids": ["2608.20318v1"], "expected_answer": "Two."},
        ]
        self.config = dict(prompt="prompt", version="codex_v1", model="test-model", k=10,
                           index_ids={"2608.20316", "2608.20318"}, index_chunks=20)
        self.records = {
            row["id"]: dict(row, k=10, model="test-model", index_chunks=20, indexed_unique_papers=2,
                            prompt_version="codex_v1", prompt_sha256=hashlib.sha256(b"prompt").hexdigest())
            for row in self.benchmark
        }

    def test_valid_partial_run_can_resume(self):
        validate_previous({"q01": self.records["q01"]}, self.benchmark, **self.config)

    def test_same_count_with_unknown_id_is_rejected(self):
        self.records["unknown"] = self.records.pop("q02")
        with self.assertRaisesRegex(SystemExit, "unknown question IDs"):
            validate_previous(self.records, self.benchmark, **self.config)

    def test_checks_later_rows_in_mixed_run(self):
        self.records["q02"]["model"] = "different-model"
        with self.assertRaisesRegex(SystemExit, "different model for q02"):
            validate_previous(self.records, self.benchmark, **self.config)

    def test_rejects_changed_question_labels_and_expected_answer(self):
        for field, value in (("question", "Changed?"), ("relevant_arxiv_ids", ["2999.00001v1"]),
                             ("expected_answer", "Changed answer.")):
            records = {qid: dict(row) for qid, row in self.records.items()}
            records["q02"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(SystemExit, f"benchmark {field} for q02"):
                validate_previous(records, self.benchmark, **self.config)

    def test_duplicate_answer_ids_are_not_silently_overwritten(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "answers.jsonl"
            path.write_text((json.dumps(self.records["q01"]) + "\n") * 2, encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "duplicate saved question ID"):
                load_previous(path)

    def test_manifest_is_immutable_and_resumes_matching_inputs(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            manifest = {"prompt": "original", "source_sha256": {"summarize.py": "abc"}}
            preserve_run_manifest(path, manifest)
            original = path.read_bytes()
            preserve_run_manifest(path, manifest, has_previous=True)
            with self.assertRaisesRegex(SystemExit, "differs"):
                preserve_run_manifest(path, dict(manifest, source_sha256={"summarize.py": "changed"}), has_previous=True)
            self.assertEqual(path.read_bytes(), original)

    def test_missing_manifest_cannot_be_added_to_existing_answers(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            with self.assertRaisesRegex(SystemExit, "no run manifest"):
                preserve_run_manifest(path, {}, has_previous=True)
            self.assertFalse(path.exists())

    def test_manifest_tracks_benchmark_bytes_and_implementation(self):
        import tempfile
        from evaluate_summaries import build_run_manifest
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "benchmark.jsonl"
            path.write_text("first benchmark", encoding="utf-8")
            config = dict(prompt="prompt", model="test-model", k=10, index_ids={"2608.20316"},
                          index_chunks=1, index_sha256="indexed passage hash")
            original = build_run_manifest(path, **config)
            path.write_text("changed benchmark", encoding="utf-8")
            changed = build_run_manifest(path, **config)
            self.assertNotEqual(original["benchmark_sha256"], changed["benchmark_sha256"])
            self.assertEqual(original["index_content_sha256"], "indexed passage hash")
            self.assertTrue({"summarize.py", "evidence.py"}.issubset(original["source_sha256"]))

    def test_index_fingerprint_detects_changed_text_and_ignores_storage_order(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from evaluate_summaries import index_content_sha256
        data = {"ids": ["a", "b"], "documents": ["one", "two"], "metadatas": [{"page": 1}, {"page": 2}]}
        collection = SimpleNamespace(metadata={"embedding_model": "test"}, get=lambda **kwargs: data)
        with patch("evaluate_summaries.index_collection", return_value=collection):
            original = index_content_sha256(Path("unused"))
            for values in data.values():
                values.reverse()
            self.assertEqual(original, index_content_sha256(Path("unused")))
            data["documents"][0] = "different passage"
            self.assertNotEqual(original, index_content_sha256(Path("unused")))

    def test_records_messages_after_historical_prompt_override(self):
        class Dummy:
            def invoke(self, messages):
                return "reply"
        recorder = RecordingLLM(Dummy())
        llm = PromptOverrideLLM(recorder, "actual prompt", clear_source_numbers=True)
        llm.invoke([("system", "old"), ("human", "[1] ID: 2608.20316v1 | Paper")])
        self.assertEqual(recorder.messages, [("system", "actual prompt"),
                                             ("human", "Source 1 | arXiv ID: 2608.20316v1 | Paper")])


class CompareSummariesCliTest(unittest.TestCase):
    def test_accepts_tuned_v3_candidate(self):
        self.check_candidate("tuned_v3")

    def test_accepts_codex_v1_with_separate_baseline_directory(self):
        self.check_candidate("codex_v1", separate_baseline=True)

    def check_candidate(self, candidate, separate_baseline=False):
        import json
        import tempfile
        from compare_summaries import main

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            baseline_out = out / "historical" if separate_baseline else out
            baseline_out.mkdir(exist_ok=True)
            summary = {
                "model": "test-model", "k": 10, "indexed_unique_papers": 50,
                "indexed_chunks": 5912, "benchmark_questions": 26,
                "mean_citation_recall": 0.75,
                "mean_citation_precision_for_answers_with_citations": 0.9,
                "answers_with_off_target_citations": 1,
            }
            for name in ("baseline", candidate):
                directory = baseline_out if name == "baseline" else out
                (directory / f"{name}_summary.json").write_text(json.dumps(summary), encoding="utf-8")
                record = {
                    "id": "q01", "question": "Question?", "relevant_arxiv_ids": ["2608.20316v1"],
                    "retrieved_arxiv_ids": ["2608.20316v1"], "citation_recall": 1,
                    "citation_precision": 1, "off_target_citation_ids": [],
                }
                (directory / f"{name}_answers.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
                (directory / f"{name}_review.csv").write_text(
                    "id,accuracy_1_to_5,business_usefulness_1_to_5\nq01,4,5\n", encoding="utf-8"
                )

            main(["--candidate", candidate, "--out", str(out), "--baseline-out", str(baseline_out)])

            comparison = out / f"prompt_comparison_{candidate}.csv"
            self.assertTrue(comparison.exists())
            text = comparison.read_text(encoding="utf-8")
            self.assertIn(f"{candidate}_citation_recall", text)
            self.assertIn(f"{candidate}_accuracy", text)
