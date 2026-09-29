"""Embed the chunks from parse_and_chunk.py and store them in a persistent ChromaDB.

    python src/parse_and_chunk.py          # first, writes data/processed/chunks.jsonl
    python src/build_index.py              # embeds and writes chroma_db/

Rebuilds the collection from scratch each run, so the index always matches the
current chunks file. Embeddings are normalized, and the collection uses cosine
distance, so the score Chroma returns is 1 - cosine similarity.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION = "kpmg1g_chunks"
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


def build(chunks_path: Path, db_dir: Path, collection_name: str, model_name: str, batch_size: int) -> int:
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

    client = chromadb.PersistentClient(path=str(db_dir))
    if collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(collection_name)
    collection = client.create_collection(
        collection_name, metadata={"hnsw:space": "cosine", "embedding_model": model_name}
    )

    # Chroma caps the batch size per add() call, so insert in slices.
    step = 1000
    for start in range(0, len(chunks), step):
        part = chunks[start : start + step]
        collection.add(
            ids=[c["chunk_id"] for c in part],
            documents=[c["text"] for c in part],
            embeddings=embeddings[start : start + step].tolist(),
            metadatas=[{k: c[k] for k in METADATA_FIELDS} for c in part],
        )
    return collection.count()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--db", type=Path, default=Path("chroma_db"))
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    count = build(args.chunks, args.db, args.collection, args.model, args.batch_size)
    log.info("Indexed %d chunks into %s (collection %s)", count, args.db, args.collection)


if __name__ == "__main__":
    main()
