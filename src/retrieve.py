"""Top-k retrieval over the ChromaDB index built by build_index.py.

    python src/retrieve.py "Which paper benchmarks agents that rewrite training algorithms?" -k 5
    python src/retrieve.py "your question" -k 8 --max-per-paper 2

When the index was built with the hub correction (the default, see build_index.py),
the query is centered with the stored corpus mean and every chunk is scored by

    score = 2 * similarity - hub_score

Search is exact: the embeddings are loaded from Chroma once and scored with one
matrix product. At this corpus size (about 6,000 chunks) that takes milliseconds
and is deterministic. Chroma's approximate (HNSW) search can skip a close chunk,
which moved MRR between 0.761 and 0.762 across runs and, worse, changed which
questions fall under the relevance floor. Past roughly a million chunks, switch
back to collection.query() with a high hnsw:search_ef.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from build_index import DEFAULT_COLLECTION, DEFAULT_MODEL, center

MAX_CANDIDATES = 1000  # top chunks turned into Hits before the per-paper cap is applied

# How many chunks summarize.py sends to the LLM, and the most from any one paper.
# 8 with at most 2 per paper guarantees at least 4 papers, the most any benchmark
# question needs. The relevance floor is checked over exactly these chunks.
CONTEXT_K = 8
CONTEXT_MAX_PER_PAPER = 2

# Below this best-chunk similarity, the question is treated as outside the corpus
# and no LLM call is made. Centered cosine, so it only applies to hub-corrected
# indexes. Set from docs/eval/retrieval_tuning_nov.md: the lowest in-scope benchmark
# question scored 0.398 and 2 of 13 out-of-scope questions scored below 0.38. It is
# deliberately conservative: most out-of-scope questions score higher and are left
# for the prompt to turn down. Re-check it with evaluate_retrieval.py --out-of-scope
# whenever the corpus or the embedding model changes.
DEFAULT_RELEVANCE_FLOOR = 0.38


@dataclass
class Hit:
    rank: int
    chunk_id: str
    arxiv_id: str
    title: str
    page_start: int
    page_end: int
    score: float  # ranking score: 2 * similarity - hub_score when hub-corrected, else the similarity
    text: str
    similarity: float = 0.0  # cosine similarity between query and chunk (centered when hub-corrected)


def select_hits(candidates: list[Hit], k: int, max_per_paper: int | None = None) -> list[Hit]:
    """Sort by score, keep at most max_per_paper chunks per paper, return the top k re-ranked from 1."""
    picked: list[Hit] = []
    per_paper: dict[str, int] = {}
    for hit in sorted(candidates, key=lambda h: h.score, reverse=True):
        if max_per_paper and per_paper.get(hit.arxiv_id, 0) >= max_per_paper:
            continue
        per_paper[hit.arxiv_id] = per_paper.get(hit.arxiv_id, 0) + 1
        picked.append(hit)
        if len(picked) == k:
            break
    for rank, hit in enumerate(picked, start=1):
        hit.rank = rank
    return picked


def best_similarity(hits: list[Hit]) -> float:
    return max((h.similarity for h in hits), default=float("-inf"))


class Retriever:
    def __init__(self, db_dir: Path = Path("chroma_db"), collection: str = DEFAULT_COLLECTION, model: str | None = None):
        import chromadb
        from sentence_transformers import SentenceTransformer

        client = chromadb.PersistentClient(path=str(db_dir))
        try:
            self.collection = client.get_collection(collection)
        except Exception as exc:  # chroma raises different types across versions
            raise SystemExit(f"No collection {collection!r} in {db_dir}. Run python src/build_index.py first.") from exc
        meta = self.collection.metadata or {}
        # Use the model the index was built with, so queries and chunks share one embedding space.
        model = model or meta.get("embedding_model", DEFAULT_MODEL)
        self.model = SentenceTransformer(model)
        self.hub_corrected = meta.get("hub_correction") == "csls"
        self.mean = np.array(json.loads(meta["embedding_mean"]), dtype=np.float32) if self.hub_corrected else None
        # The floor was calibrated on centered similarities, so a plain index gets none.
        self.relevance_floor = DEFAULT_RELEVANCE_FLOOR if self.hub_corrected else None

        data = self.collection.get(include=["embeddings", "documents", "metadatas"])
        if not data["ids"]:
            raise SystemExit(f"Collection {collection!r} in {db_dir} is empty. Run python src/build_index.py.")
        self.ids = data["ids"]
        self.documents = data["documents"]
        self.metadatas = data["metadatas"]
        self.embeddings = np.asarray(data["embeddings"], dtype=np.float32)  # unit-length rows
        hubs = [m.get("hub_score", 0.0) for m in self.metadatas]
        self.hub_scores = np.asarray(hubs, dtype=np.float32) if self.hub_corrected else np.zeros(len(self.ids), np.float32)

    def count(self) -> int:
        return len(self.ids)

    def embed_query(self, query: str) -> list[float]:
        vector = self.model.encode([query], normalize_embeddings=True)
        if self.hub_corrected:
            vector = center(vector, self.mean)
        return vector[0].tolist()

    def search(self, query: str, k: int = 5, max_per_paper: int | None = None) -> list[Hit]:
        similarities = self.embeddings @ np.asarray(self.embed_query(query), dtype=np.float32)
        scores = 2 * similarities - self.hub_scores if self.hub_corrected else similarities
        n = min(len(scores), max(MAX_CANDIDATES, k))
        top = np.argpartition(-scores, n - 1)[:n] if n < len(scores) else np.arange(len(scores))
        candidates = []
        for i in top:
            meta = self.metadatas[i]
            candidates.append(
                Hit(
                    rank=0,
                    chunk_id=self.ids[i],
                    arxiv_id=meta["arxiv_id"],
                    title=meta["title"],
                    page_start=meta["page_start"],
                    page_end=meta["page_end"],
                    score=float(scores[i]),
                    text=self.documents[i],
                    similarity=float(similarities[i]),
                )
            )
        return select_hits(candidates, k, max_per_paper)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--max-per-paper", type=int, default=None, help="cap chunks per paper so more papers show up")
    parser.add_argument("--db", type=Path, default=Path("chroma_db"))
    args = parser.parse_args(argv)
    if args.k <= 0:
        parser.error("-k must be positive")
    if args.max_per_paper is not None and args.max_per_paper <= 0:
        parser.error("--max-per-paper must be positive")

    retriever = Retriever(args.db)
    hits = retriever.search(args.query, args.k, args.max_per_paper)
    for hit in hits:
        print(f"{hit.rank}. [{hit.score:.3f}] {hit.arxiv_id} p.{hit.page_start}-{hit.page_end}  {hit.title}")
        print(f"   {hit.text[:200]}...")
    if retriever.relevance_floor is not None and best_similarity(hits) < retriever.relevance_floor:
        print(f"\nBest similarity {best_similarity(hits):.3f} is below {retriever.relevance_floor}: likely outside the corpus.")


if __name__ == "__main__":
    main()
