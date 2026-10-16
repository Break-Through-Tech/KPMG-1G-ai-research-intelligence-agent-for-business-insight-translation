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
| `expected_answer` | what a correct answer should say, written from the abstract. Used to score generated answers with the rubric below |

Extra fields (for example `rubric_accuracy`, `rubric_usefulness`, `notes`) are ignored by the retrieval eval, so the rubric can live in the same file.

## Files

- `questions.jsonl`: **the real benchmark (Issue #5, Sarah).** 26 questions: 20 single-paper and 6 multi-paper (q21–q26). Frozen as of 10/15. Don't edit it while tuning; if it needs changes, bump to a new version and note why.
- - `smoke_questions.jsonl`: 10 throwaway questions, two per seeded paper, written from the abstracts only to check the pipeline runs end to end. They lean on abstract wording, so they overstate how well retrieval works. Do not report them as the baseline.
 
## Scoring rubric (for generated answers)

Score each generated answer against the question's `expected_answer` and the source paper(s).

**Accuracy (1–5)**: are the findings correct?
- 5: all key claims and numbers correct, nothing invented
- 3: main idea right, but a number is off or a key finding is missing
- 1: wrong paper, wrong findings, or made-up claims

**Business usefulness (1–5)**: could a KPMG analyst act on it?
- 5: clear plain-language takeaway, says why it matters and for whom
- 3: accurate but reads like a research summary, implication is vague
- 1: jargon-heavy or no business takeaway
