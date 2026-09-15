# AI Research Intelligence Agent for Business Insight Translation

A retrieval-augmented generation (RAG) system over public AI research from arXiv. Given a
natural-language question, it retrieves the relevant papers, summarizes the findings, and
translates them into business-relevant implications with citations back to the source papers.

**Host company:** KPMG
**Program:** Break Through Tech AI Studio, Fall 2026 (Team KPMG 1G)
**Challenge Advisor:** Abhinav Raghunathan, KPMG
**AI Studio Coach:** Alexandra Ladyzhensky, Break Through Tech

---

### Team Members

| Name | GitHub Handle | Contribution |
|---|---|---|
| Arsenii Chan | @ArseniiChan | Repo and environment setup, parsing and chunking pipeline, baseline RAG pipeline |
| Sarah Khadder | @skhadder | Evaluation framework: benchmark queries and scoring rubrics |
| Amy Weston | @amyweston | Exploratory data analysis, baseline RAG pipeline |
| Nathanielle Onchengco | _pending org invite_ | arXiv corpus collection and metadata |

---

## Project Overview

KPMG needs a faster way to monitor the volume of AI research being published and turn it into
something a business audience can act on. Reading and triaging arXiv by hand does not scale.

This project builds a prototype that does three things:

1. **Retrieve** the papers relevant to a natural-language query
2. **Summarize** what each paper actually found
3. **Translate** those findings into business implications, with citations

Scope is a working prototype with a documented evaluation approach. Out of scope: production
deployment, fine-tuning foundation models, autonomous multi-agent orchestration, and any use of
PII, regulated, internal, or client data.

**Success criteria (set by the Challenge Advisor):** retrieval relevance, summary accuracy and
clarity, business usefulness of the translated output, and human validation by KPMG stakeholders.

---

## Milestones

| Month | Milestone | Status |
|---|---|---|
| September | Foundation and baseline RAG: ingest the arXiv corpus, define the evaluation framework, stand up a baseline retrieval pipeline | In progress |
| October | Pipeline development and evaluation: retrieval + summarization, prompt engineering for business translation | Not started |
| November | Refinement, interaction layer, documentation and final delivery | Not started |

Tasks are tracked as GitHub Issues against the matching Milestone and on our
[GitHub Project board](https://github.com/orgs/Break-Through-Tech/projects/224).

---

## Repository Structure

```
.
├── data/                 # arXiv PDFs and metadata
├── notebooks/            # EDA and experiments
└── README.md
```

`src/` for reusable ingestion, retrieval and evaluation code, and `data/processed/` for parsed
output, will be added alongside the first pipeline code.

---

## Setup and Installation

**1. Clone the repository**

```bash
git clone https://github.com/Break-Through-Tech/KPMG-1G-ai-research-intelligence-agent-for-business-insight-translation.git
cd KPMG-1G-ai-research-intelligence-agent-for-business-insight-translation
```

**2. Create a virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

The dependency list is still being settled with our AI Studio Coach, so `requirements.txt` is not
final yet. Until it is, install per-notebook in Colab.

**3. Environment variables**

API keys go in a local `.env` file. Never commit a key.

**4. Data**

Five seeded arXiv cs.AI papers are already in `data/`. The full corpus is pulled by the
ingestion script (see Issue #2) rather than committed by hand.

**5. Shared Colab notebook**

Shared team notebook: [KPMG1G_baseline_rag.ipynb](https://colab.research.google.com/drive/1XHMbyuiAU2lzfe9NQBzN-Uz2XzLwgiBT)

Shared Drive folder: [KPMG 1G — AI Studio Fall 2026](https://drive.google.com/drive/folders/1AGX5d8Osa8sMA0KQxRUkUPwzUZWLAVgg)

---

## Data Exploration

Dataset: recent papers from [arXiv cs.AI](https://arxiv.org/list/cs.AI/recent), PDF plus
metadata (id, title, authors, date, abstract, categories). Under 1 GB.

EDA is tracked in Issue #3 and will cover length distribution, section structure, per-paper parse
failure rate, and metadata gaps. Findings and visualizations will be added here once that work
lands.

---

## Model Development

Baseline stack, per the Challenge Advisor's recommended tooling:

| Stage | Choice |
|---|---|
| PDF parsing | `pypdf` |
| Embeddings | `sentence-transformers` |
| Vector store | `ChromaDB` |
| Orchestration | `LangChain` |
| Generation | Free-tier LLM API |

Chunking strategy will be set from the EDA results rather than guessed. Details to follow once the
baseline is running.

---

## Results and Key Findings

Not yet available. The baseline retrieval score is due at the end of September (Issue #5) and will
be recorded here with the evaluation method alongside it.

Planned metrics: Recall@k for retrieval relevance, and 1-5 team rubrics for summary accuracy and
business usefulness.

---

## Next Steps

- Finish the September milestone: corpus ingestion, EDA, chunking pipeline, baseline RAG
- Record the baseline evaluation score before adding any improvements
- Confirm target paper count and benchmark queries with the Challenge Advisor

---

## License

To be selected with the Challenge Advisor's approval.

---

## Acknowledgements

Thank you to Abhinav Raghunathan (KPMG) for advising this project, and to Alexandra Ladyzhensky
and the Break Through Tech AI Studio team for their support.
