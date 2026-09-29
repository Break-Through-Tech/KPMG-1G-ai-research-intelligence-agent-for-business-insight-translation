# Benchmark questions

`src/evaluate_retrieval.py` scores retrieval against a JSONL file with one question per line:

```json
{"id": "q01", "question": "...", "relevant_arxiv_ids": ["2608.20316v1"]}
```

| field | meaning |
|---|---|
| `id` | short unique id |
| `question` | phrased the way a KPMG analyst would ask it |
| `relevant_arxiv_ids` | the paper(s) that should come back, matching the ids in `data/` filenames |

Extra fields (for example `rubric_accuracy`, `rubric_usefulness`, `notes`) are ignored by the retrieval eval, so the rubric can live in the same file.

## Files

- `questions.jsonl`: **the real benchmark (Issue #5, Sarah).** Not written yet. Freeze it before we tune anything.
- `smoke_questions.jsonl`: 10 throwaway questions, two per seeded paper, written from the abstracts only to check the pipeline runs end to end. They lean on abstract wording, so they overstate how well retrieval works. Do not report them as the baseline.
