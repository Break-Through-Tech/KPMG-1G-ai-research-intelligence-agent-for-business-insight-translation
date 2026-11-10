"""Embed the chunks from parse_and_chunk.py and store them in a persistent ChromaDB.

    python src/parse_and_chunk.py          # first, writes data/processed/chunks.jsonl
    python src/build_index.py              # embeds and writes chroma_db/
    python src/build_index.py --no-hub-correction   # plain embeddings, for comparison

Rebuilds the collection from scratch each run, so the index always matches the
current chunks file. Embeddings are normalized, and the collection uses cosine
distance, so the score Chroma returns is 1 - cosine similarity.

Hub correction (on by default)
------------------------------
With all-MiniLM-L6-v2, a few papers show up near the top for almost every question.
On the Oct 27 baseline, 2610.18842v1 was in the top 5 for 14 of 26 benchmark
questions. Their generic "AI systems are changing" passages sit close to every
query in embedding space. Two standard fixes are applied at build time:

  1. Centering: subtract the mean chunk embedding from every chunk (and later from
     every query) and re-normalize. This removes the direction that all AI papers
     share, so similarity reflects what is specific to a passage.
  2. CSLS (Conneau et al., 2018, "Word Translation Without Parallel Data"): each
     chunk gets a hub score, its mean similarity to its k nearest reference queries.
     retrieve.py ranks by 2 * similarity - hub_score, so a chunk that is close to
     every question is pushed down and a chunk that is close to this question only
     is not.

The reference queries live in data/reference_queries.txt: the Challenge Advisor's
46 sample user prompts. They must not include benchmark questions.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION = "kpmg1g_chunks"
DEFAULT_REFERENCE = Path("data/reference_queries.txt")
HUB_K = 10  # neighbors per chunk for the hub score, the default from the CSLS paper
METADATA_FIELDS = ("arxiv_id", "title", "source_file", "section", "page_start", "page_end", "chunk_index")

log = logging.getLogger("build_index")


def load_chunks(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path} not found. Run python src/parse_and_chunk.py first.")
    with open(path, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    if not chunks:
        raise SystemExit(f"{path} is empty.")
    ids = [c["chunk_id"] for c in chunks]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"{path} has duplicate chunk_ids.")
    return chunks


def load_reference_queries(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"{path} not found. Pass --reference or --no-hub-correction.")
    with open(path, encoding="utf-8") as f:
        queries = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    if not queries:
        raise SystemExit(f"{path} has no queries.")
    return queries


def center(vectors: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Subtract the corpus mean and re-normalize each row to unit length."""
    shifted = np.asarray(vectors, dtype=np.float32) - mean
    norms = np.linalg.norm(shifted, axis=1, keepdims=True)
    return shifted / np.maximum(norms, 1e-12)


def hub_scores(chunk_vectors: np.ndarray, reference_vectors: np.ndarray, k: int) -> np.ndarray:
    """Per chunk: mean cosine similarity to its k most similar reference queries.

    Both inputs are unit-length rows, so the dot product is the cosine similarity.
    """
    k = max(1, min(k, len(reference_vectors)))
    sims = reference_vectors @ chunk_vectors.T  # (n_reference, n_chunks)
    return np.sort(sims, axis=0)[-k:].mean(axis=0)


def build(
    chunks_path: Path,
    db_dir: Path,
    collection_name: str,
    model_name: str,
    batch_size: int,
    reference_path: Path | None = DEFAULT_REFERENCE,
) -> int:
    """reference_path=None builds a plain index with no hub correction."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    chunks = load_chunks(chunks_path)
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        [c["text"] for c in chunks],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    collection_meta = {"hnsw:space": "cosine", "embedding_model": model_name, "hub_correction": "none"}
    hubs = np.zeros(len(chunks), dtype=np.float32)
    if reference_path is not None:
        references = load_reference_queries(reference_path)
        mean = embeddings.mean(axis=0)
        embeddings = center(embeddings, mean)
        reference_vectors = center(model.encode(references, normalize_embeddings=True), mean)
        hubs = hub_scores(embeddings, reference_vectors, HUB_K)
        collection_meta.update(
            hub_correction="csls",
            hub_k=min(HUB_K, len(references)),
            reference_queries=str(reference_path),
            reference_count=len(references),
            # Queries must be centered with the same mean, so it travels with the index.
            embedding_mean=json.dumps([round(float(x), 7) for x in mean]),
        )

    client = chromadb.PersistentClient(path=str(db_dir))
    if collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(collection_name)
    collection = client.create_collection(collection_name, metadata=collection_meta)

    # Chroma caps the batch size per add() call, so insert in slices.
    step = 1000
    for start in range(0, len(chunks), step):
        part = chunks[start : start + step]
        collection.add(
            ids=[c["chunk_id"] for c in part],
            documents=[c["text"] for c in part],
            embeddings=embeddings[start : start + step].tolist(),
            metadatas=[
                {**{k: c[k] for k in METADATA_FIELDS}, "hub_score": float(hubs[start + i])}
                for i, c in enumerate(part)
            ],
        )
    return collection.count()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--db", type=Path, default=Path("chroma_db"))
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE, help="reference queries for the hub correction")
    parser.add_argument("--no-hub-correction", action="store_true", help="store plain embeddings (the Oct 27 baseline setup)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    reference = None if args.no_hub_correction else args.reference
    count = build(args.chunks, args.db, args.collection, args.model, args.batch_size, reference)
    mode = "no hub correction" if reference is None else f"hub correction from {reference}"
    log.info("Indexed %d chunks into %s (collection %s, %s)", count, args.db, args.collection, mode)


if __name__ == "__main__":
    main()
