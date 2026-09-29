"""Top-k retrieval over the ChromaDB index built by build_index.py.

    python src/retrieve.py "Which paper benchmarks agents that rewrite training algorithms?" -k 5
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from build_index import DEFAULT_COLLECTION, DEFAULT_MODEL


@dataclass
class Hit:
    rank: int
    chunk_id: str
    arxiv_id: str
    title: str
    page_start: int
    page_end: int
    score: float  # cosine similarity, higher is closer
    text: str


class Retriever:
    def __init__(self, db_dir: Path = Path("chroma_db"), collection: str = DEFAULT_COLLECTION, model: str | None = None):
        import chromadb
        from sentence_transformers import SentenceTransformer

        client = chromadb.PersistentClient(path=str(db_dir))
        try:
            self.collection = client.get_collection(collection)
        except Exception as exc:  # chroma raises different types across versions
            raise SystemExit(f"No collection {collection!r} in {db_dir}. Run python src/build_index.py first.") from exc
        # Use the model the index was built with, so queries and chunks share one embedding space.
        model = model or (self.collection.metadata or {}).get("embedding_model", DEFAULT_MODEL)
        self.model = SentenceTransformer(model)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        embedding = self.model.encode([query], normalize_embeddings=True).tolist()
        res = self.collection.query(query_embeddings=embedding, n_results=k)
        hits = []
        for rank, (cid, doc, meta, dist) in enumerate(
            zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]), start=1
        ):
            hits.append(
                Hit(
                    rank=rank,
                    chunk_id=cid,
                    arxiv_id=meta["arxiv_id"],
                    title=meta["title"],
                    page_start=meta["page_start"],
                    page_end=meta["page_end"],
                    score=1 - dist,
                    text=doc,
                )
            )
        return hits


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--db", type=Path, default=Path("chroma_db"))
    args = parser.parse_args(argv)
    if args.k <= 0:
        parser.error("-k must be positive")

    for hit in Retriever(args.db).search(args.query, args.k):
        print(f"{hit.rank}. [{hit.score:.3f}] {hit.arxiv_id} p.{hit.page_start}-{hit.page_end}  {hit.title}")
        print(f"   {hit.text[:200]}...")


if __name__ == "__main__":
    main()
