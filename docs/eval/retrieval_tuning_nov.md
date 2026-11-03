# Retrieval tuning, November (Issue #15)

Issue #15: refine prompts, retrieval ranking and relevance mechanisms based on the evaluation
results. This file records what was tried, what was kept, what was dropped and why, so the
numbers in the README can be traced back.

## Result

Benchmark: the frozen 26 questions in `data/benchmark/questions.jsonl` (20 single-paper,
6 multi-paper). Corpus: the 50 papers in `data/metadata.csv`. Same embedding model as the
baseline (`all-MiniLM-L6-v2`).

| | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|
| Oct 27 baseline | 0.542 | 0.814 | 0.846 | 0.761 |
| Now | 0.696 | 0.853 | 0.904 | 0.854 |

Relevant papers that actually reach the LLM in the passages it is given:

| | single-paper questions | multi-paper questions |
|---|---|---|
| Oct 27 (5 chunks, no per-paper limit) | 0.900 | 0.278 |
| Now (8 chunks, at most 2 per paper) | 1.000 | 0.625 |

What changed:

1. **Hub correction** in `build_index.py` / `retrieve.py`. Embeddings are centered on the corpus
   mean, and each chunk is penalized by how close it sits to typical user questions (CSLS).
   No new model and no extra query time.
2. **Wider, more varied context** for the LLM in `summarize.py`: 8 chunks, at most 2 from any
   one paper.
3. **Parser fallback** in `parse_and_chunk.py`: pages where pypdf drops the spaces between words
   are re-read with pypdfium2. G-CARL (2608.20331v1) goes from 25 usable chunks to 49.
4. **Two-layer "not covered" handling**: a conservative retrieval floor that skips the LLM call,
   plus a prompt that tells the model when and how to decline.
5. **Prompt v2** in `summarize.py`, built from the answer pattern and stress tests in the
   Challenge Advisor's sample-prompts document. The October prompt is kept as `--prompt v1` so
   the two can be scored side by side.

## How the experiments were run

- Every configuration was scored with the metric functions in `evaluate_retrieval.py`, using exact
  search over cached embeddings so that approximate-search noise did not blur small differences.
  Exact search reproduces the Chroma baseline to three decimals (MRR 0.761).
- The benchmark is frozen and small. 0.02 of MRR is about half a question. To avoid tuning to
  the test set, a change was kept only if it (a) improved both halves of the benchmark (odd and
  even question ids) and (b) did not trade away Recall@5 for Recall@1.
- Parameters were set from published defaults or a stated rule, not by picking the best score
  (see the CSLS `k` below).
- The 10 smoke questions are reported but not used for decisions. They are written from the
  abstracts and overstate retrieval quality.

## Every configuration tried

| Config | R@1 | R@3 | R@5 | MRR | MRR odd / even | Notes |
|---|---|---|---|---|---|---|
| Baseline, MiniLM dense | 0.542 | 0.814 | 0.846 | 0.761 | 0.885 / 0.638 | |
| + parser fallback | 0.542 | 0.814 | 0.846 | 0.761 | 0.885 / 0.638 | kept: no score change, 24 more G-CARL chunks |
| BM25 only | 0.503 | 0.689 | 0.756 | 0.699 | 0.737 / 0.662 | |
| MiniLM + BM25 (reciprocal rank fusion) | 0.667 | 0.795 | 0.814 | 0.832 | 0.891 / 0.774 | dropped: R@5 falls |
| MiniLM, cross-encoder rerank of top 50 | 0.628 | 0.843 | 0.885 | 0.805 | 0.872 / 0.738 | dropped: 4.7 s per query |
| MiniLM + BM25, cross-encoder rerank | 0.590 | 0.785 | 0.795 | 0.779 | 0.824 / 0.735 | dropped: worse than without it |
| bge-small-en-v1.5 dense | 0.619 | 0.747 | 0.843 | 0.774 | 0.891 / 0.657 | dropped: no gain over MiniLM |
| MiniLM, title prepended to each chunk | 0.657 | 0.785 | 0.843 | 0.815 | 0.846 / 0.785 | dropped: odd half falls |
| Abstracts only (title + abstract per paper) | 0.667 | 0.824 | 0.833 | 0.852 | 0.888 / 0.816 | dropped: see note |
| Chunks hybrid + abstracts (fused) | 0.657 | 0.833 | 0.843 | 0.828 | 0.892 / 0.763 | dropped: no gain over hybrid |
| MiniLM, centered only | 0.599 | 0.776 | 0.846 | 0.763 | 0.923 / 0.604 | |
| MiniLM, CSLS only | 0.657 | 0.862 | 0.904 | 0.832 | 0.930 / 0.733 | |
| **MiniLM, centered + CSLS** | **0.696** | **0.853** | **0.904** | **0.854** | **0.923 / 0.785** | **kept** |
| MiniLM + BM25, centered | 0.705 | 0.785 | 0.785 | 0.856 | 0.928 / 0.784 | dropped: R@5 falls to 0.785 |

Notes:

- **Cross-encoder** (`cross-encoder/ms-marco-MiniLM-L6-v2`, the reranker the Challenge Overview
  suggests). It is trained on web-search queries. Our questions are business-phrased and the
  passages are technical, and it ranked worse after hybrid retrieval. `BAAI/bge-reranker-base`
  was too slow on CPU to be practical (5.6 s per query).
- **Abstracts only** scores well, but the benchmark's expected answers were written from the
  abstracts, so part of that score is the benchmark rewarding its own source. It is not a fair
  test, so it was not adopted.

## Why the hub correction works here

Some passages sit close to every question in embedding space. For all-MiniLM-L6-v2 on this
corpus, 2610.18842v1 (a survey on robots improving themselves) was in the top 5 for 14 of the
26 benchmark questions, and 2610.35802v1 (game theory of academic hiring) showed up for the
healthcare question. Their generic "AI is changing how X works" prose matches everything.

- **Centering** subtracts the mean chunk embedding, which removes the direction all AI papers
  share.
- **CSLS** (Conneau et al., 2018) gives each chunk a hub score: its mean similarity to the 10
  reference queries it is closest to. Ranking by `2 * similarity - hub_score` pushes down
  passages that are close to every question, and leaves alone passages that are close to only
  this one.
- The reference queries are the Challenge Advisor's 46 sample prompts
  (`data/reference_queries.txt`). They describe what real users ask. None of them is a benchmark
  question (a unit test checks for identical text), and the closest pair by embedding is only
  0.648 cosine (q25 "progress in healthcare and life sciences" against "research areas that could
  affect audit, tax, advisory").
- 7 of the 13 out-of-scope questions below are also advisor prompts, so they are in the reference
  set. Recomputing the hub scores without them leaves every out-of-scope similarity unchanged to
  three decimals, so that overlap does not flatter the out-of-scope check.

Sensitivity checks:

| Check | MRR | R@5 |
|---|---|---|
| CSLS k = 5 / 10 / 20 / 40 | 0.873 / 0.854 / 0.835 / 0.769 | 0.904 at every k |
| 5 random halves (23 prompts) of the reference set | 0.790 to 0.856 | 0.894 to 0.904 |
| Reference set = the 10 smoke questions instead | 0.768 | 0.853 |
| Same correction on bge-small-en-v1.5 | 0.780 (vs 0.774 plain) | 0.853 |
| Score only the top 100 / 200 / 400 candidates | 0.854 at every pool size | 0.904 |

- k = 5 scores higher than 10, but 10 is the CSLS paper's default and was fixed before looking.
  Picking the best k on this benchmark would be tuning to the test set.
- The gain depends on the reference set looking like real usage. With 10 narrow, abstract-worded
  questions it disappears. With half of the advisor's prompts it shrinks but stays above the
  baseline.
- The gain is specific to MiniLM's hub problem. If the team changes the embedding model, rerun the
  eval with `--no-hub-correction` and without before keeping it.
- Search is exact: `retrieve.py` loads the embeddings from Chroma once and scores every chunk
  with one matrix product (milliseconds at this size). An earlier version re-ranked Chroma's
  approximate top 100. That gave the same paper ranking, but HNSW sometimes skipped a close chunk,
  which changed the relevance-floor result for one out-of-scope question. Exact search also removes
  the 0.761 / 0.762 MRR wobble noted for the Oct 27 baseline.

## Getting more papers in front of the LLM

`summarize.py` used to send the top 5 chunks. On multi-paper questions those 5 chunks often came
from one or two papers, so the model could not compare papers it never saw. Share of relevant
papers present in the chunks sent:

| Retrieval | Chunks | Max per paper | Single | Multi |
|---|---|---|---|---|
| Baseline | 5 | none | 0.900 | 0.278 |
| Hub-corrected | 5 | none | 1.000 | 0.278 |
| Hub-corrected | 5 | 2 | 1.000 | 0.444 |
| **Hub-corrected** | **8** | **2** | **1.000** | **0.625** |
| Hub-corrected | 8 | 1 | 1.000 | 0.708 |

8 chunks with at most 2 per paper guarantees at least 4 papers, which is the most any benchmark
question needs, while still giving the model two passages per paper to pull numbers from. One
per paper covers more papers but leaves a single passage of evidence for each. The added prompt
length is about 400 words.

## Saying "not covered"

The advisor's stress tests include "ask about a topic that is absent from the corpus and verify
that the agent clearly reports the gap." `data/benchmark/out_of_scope.jsonl` has 13 questions the
corpus cannot answer: 7 taken verbatim from the advisor's sample prompts and 6 written for this
test. Each line records why it is out of scope, checked against all 50 abstracts. Five are hard
negatives that share words with papers in the corpus (for example "chunking strategies for RAG"
against the robot action-chunking papers).

Can a retrieval score alone tell them apart from the 36 in-scope questions (benchmark + smoke)?
Best similarity over all chunks, from the experiment harness (run before the soft-hyphen parser
fix; the floor numbers below come from the final rebuilt index):

| Score | Lowest in-scope | Out-of-scope median / max | Caught at the lowest in-scope score |
|---|---|---|---|
| Cosine similarity (plain) | 0.449 | 0.487 / 0.588 | 3 of 13 |
| Cosine similarity (centered) | 0.399 | 0.440 / 0.560 | 3 of 13 |
| ms-marco cross-encoder | -7.36 | -4.45 / -0.03 | 3 of 13 (+0.7 s per query) |
| bge-reranker-base | 0.00 | 0.01 / 0.12 | 0 of 13 (+5.6 s per query) |

No. Some real questions score low because business wording does not match technical text, and
some out-of-scope questions score high because a neighboring paper exists (synthetic data,
prompt injection). A hard threshold either rejects real questions or lets most out-of-scope ones
through. So it is two layers:

1. **Retrieval floor** (`retrieve.DEFAULT_RELEVANCE_FLOOR = 0.38`, centered similarity). Below it
   the question is clearly off-topic and no LLM call is spent. It is checked over the same 8
   chunks `summarize.py` sends with its default settings (`-k` and `--max-per-paper` change
   that). On this corpus it catches 2 of 13 (self-driving cars at 0.368, federated learning at
   0.372) and stops none of the 26 benchmark questions. The closest calls are q10 at 0.398 on
   the in-scope side and the RAG-hallucination question at 0.382 on the out-of-scope side.
2. **Prompt.** Every other question goes to the model with instructions to say
   "The retrieved papers do not cover this question." when no passage addresses it, to flag
   partial coverage, and to correct false premises.

`python src/evaluate_retrieval.py --out-of-scope data/benchmark/out_of_scope.jsonl` prints the
margin between the floor and the lowest in-scope question, so it can be re-checked whenever the
corpus changes.

## Prompt v2

v1 asked for 3 to 6 plain sentences with citations. The advisor's sample-prompts document asks
for a specific answer pattern and lists stress tests. v2 maps to them one to one:

| Advisor's document | Prompt v2 |
|---|---|
| (1) concise answer | **Answer:** 2-3 sentences |
| (2) most relevant papers and why they matter, (3) key findings | **Evidence:** one bullet per paper, with its numbers and [arXiv:ID] |
| (3) limitations | **Limitations:** benchmark-only, one study, small scale, narrow setting |
| (4) business implications | **Business implications:** worded as interpretation, not findings |
| (5) citations or links to source metadata | [arXiv:ID] on every finding; sources list prints title, first author, date, arXiv link |
| absent topic | exact "not covered" sentence plus the closest topic the papers do cover |
| false premise | correct it first |
| unsupported conclusion | say so instead of giving it |
| ambiguous term | state the reading used |
| "do not treat repeated claims as independent evidence" | two passages from one paper are not independent evidence |

**Not yet measured.** There is no Gemini key in this environment, so v2 has unit tests but no
generated answers. To score it against v1 on the team's 1-5 rubrics:

```bash
python src/summarize.py --batch data/benchmark/questions.jsonl --prompt v1
python src/summarize.py --batch data/benchmark/questions.jsonl --prompt v2
python src/summarize.py --batch data/benchmark/out_of_scope.jsonl --prompt v2
```

Each run writes `data/processed/answers/<file>_<prompt>.jsonl` with the answer, the papers
retrieved and cited, `relevant_cited`, `abstained`, and empty `rubric_accuracy` /
`rubric_usefulness` fields to score by hand. For the out-of-scope file, every row should have
`abstained: true`.

## Benchmark label gaps (for Sarah to review)

The answer keys were written when the benchmark covered the 20 seeded papers. On Oct 27 the
corpus grew to 50, and some of the 30 new papers answer the broad questions too. Reading their
abstracts, these look like missing labels:

| Question | Proposed addition | Strength | Why |
|---|---|---|---|
| q10 offline learning from logs | 2610.35656v1 | partial | sample-efficient Q-learning theory, but online robust RL |
| q22 VLA reliability | 2610.35832v1 | strong | evaluates a frontier agent against VLA policies on manipulation |
| q22 | 2610.35817v1 | partial | sim-trained agent compared with a VLA |
| q23 cutting inference cost | 2610.35690v1 | strong | "cost-efficient multi-agent routing" |
| q23 | 2610.35804v1 | strong | faster decoding for diffusion language models |
| q24 safeguards for self-modifying AI | 2610.35835v1 | partial | enforcing human approval of agent actions |
| q25 healthcare and life sciences | 2610.35704v1, 2610.35721v1, 2610.35761v1, 2610.35847v1 | strong | breast biopsy, clinical-trial screening, mental-health risk, cancer-care evaluation |
| q26 reward design for RL on LLMs | 2610.35716v1 | strong | auditing reward hacking in GRPO |
| q26 | 2610.35697v1, 2610.35733v1 | partial | error-targeted RL; multi-objective RL for post-training |

These are proposals, not edits. The benchmark stays frozen and every score above uses the
original labels. With the strong additions, the three main configurations rank in the same
order, so the conclusions do not depend on the gaps:

| | R@5 | MRR |
|---|---|---|
| Baseline | 0.862 | 0.769 |
| MiniLM + BM25 | 0.824 | 0.848 |
| Centered + CSLS | 0.919 | 0.878 |

If the team agrees, the clean way to apply them is a `questions_v2.jsonl` with a note, as
`data/benchmark/README.md` already asks.

## What will break first, and what to do next

| Breaking point | Sign | Next step |
|---|---|---|
| Corpus grows or changes topic | hub penalty drifts, floor margin shrinks | rebuild the index (hub scores are recomputed), rerun `--out-of-scope` |
| Embedding model changes | correction may stop helping (it did nothing for bge-small) | compare with `--no-hub-correction` before keeping it |
| Reference prompts do not match real usage | gain shrinks toward baseline | add real user questions from testing to `data/reference_queries.txt`, never benchmark ones |
| Floor margin is thin (0.38 vs 0.398 for q10) | an in-scope question gets "not covered" without an LLM call | lower the floor or switch it off; the prompt still handles out-of-scope questions |
| Business wording vs technical text (q08 and q20 still rank the right paper 2nd) | a near-miss paper ranks first | query rewriting: have the LLM turn the question into technical search terms before retrieval. Needs a key to test. |
| Some reference lists survive parsing (EvoRouter, 2610.35690v1, pages 9 to 10 are kept as "appendix") | citation-only chunks show up in results | tighten `remove_back_matter()` in `parse_and_chunk.py`; not part of this change |
| 26 questions is a small benchmark | differences under ~0.03 MRR are noise | grow it with the questions from user testing (Issue #17) |

## Reproduce

```bash
python src/pull_arxiv.py --ids-from data/metadata.csv
python src/parse_and_chunk.py
python src/build_index.py --no-hub-correction --db chroma_db_baseline
python src/evaluate_retrieval.py --db chroma_db_baseline --out data/processed/eval_baseline
python src/build_index.py
python src/evaluate_retrieval.py --out-of-scope data/benchmark/out_of_scope.jsonl
python src/evaluate_retrieval.py --benchmark data/benchmark/smoke_questions.jsonl
python src/plot_eval.py --baseline-dir data/processed/eval_baseline
```
