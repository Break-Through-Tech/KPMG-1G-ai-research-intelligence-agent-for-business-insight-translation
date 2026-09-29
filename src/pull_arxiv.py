"""Pull recent arXiv cs.AI papers into data/ and keep a metadata file next to them.

    python src/pull_arxiv.py                      # newest 50 cs.AI papers
    python src/pull_arxiv.py --max 200            # more
    python src/pull_arxiv.py --metadata-only      # refresh metadata.csv, no PDF downloads

Safe to rerun: papers already in data/ (matched by arXiv id, any version) are not
downloaded again, and data/metadata.csv is rewritten to cover every PDF in data/,
including the seeded ones. PDFs are saved as "<id>_<title>.pdf", the name format
parse_and_chunk.py reads the id and title from.

arXiv asks for no more than one request every 3 seconds. The API client waits
that long between pages and retries on errors, and PDF downloads pause 3 seconds.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import time
import urllib.request
from pathlib import Path

log = logging.getLogger("pull_arxiv")

ID_IN_FILENAME_RE = re.compile(r"^(?P<base>\d{4}\.\d{4,5})(?P<version>v\d+)_")
METADATA_FIELDS = ["arxiv_id", "title", "authors", "published", "updated", "primary_category", "categories", "abstract", "pdf_file"]


def safe_filename(arxiv_id: str, title: str, max_title: int = 80) -> str:
    """'2608.20316v1' + 'Pandora's Box: Routing' -> "2608.20316v1_Pandora's Box Routing.pdf"."""
    clean = re.sub(r"[\\/:*?\"<>|\n\r\t]+", " ", title)
    clean = re.sub(r"\s+", " ", clean).strip()[:max_title].rstrip(" .")
    return f"{arxiv_id}_{clean}.pdf"


def existing_papers(data_dir: Path) -> dict[str, Path]:
    """Map base arXiv id (no version) -> PDF already on disk."""
    found = {}
    for pdf in data_dir.glob("*.pdf"):
        m = ID_IN_FILENAME_RE.match(pdf.name)
        if m:
            found[m["base"]] = pdf
    return found


def base_id(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id)


def pdf_url(result) -> str:
    url = getattr(result, "pdf_url", None)
    if not url:
        url = next((l.href for l in result.links if getattr(l, "title", None) == "pdf"), None)
    return url or f"https://arxiv.org/pdf/{result.get_short_id()}"


def download(url: str, dest: Path) -> None:
    """Stream a PDF to disk. Written to a .part file first so a failed download leaves nothing behind."""
    tmp = dest.with_suffix(".part")
    req = urllib.request.Request(url, headers={"User-Agent": "kpmg1g-btt-ai-studio/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
        while chunk := resp.read(1 << 16):
            out.write(chunk)
    with open(tmp, "rb") as f:
        if f.read(5) != b"%PDF-":
            tmp.unlink()
            raise ValueError("response was not a PDF")
    tmp.rename(dest)


def to_row(result, pdf_file: str) -> dict:
    return {
        "arxiv_id": result.get_short_id(),
        "title": re.sub(r"\s+", " ", result.title).strip(),
        "authors": "; ".join(a.name for a in result.authors),
        "published": result.published.date().isoformat(),
        "updated": result.updated.date().isoformat(),
        "primary_category": result.primary_category,
        "categories": ";".join(result.categories),
        "abstract": re.sub(r"\s+", " ", result.summary).strip(),
        "pdf_file": pdf_file,
    }


def pull(data_dir: Path, category: str, max_results: int, metadata_only: bool) -> list[dict]:
    import arxiv

    data_dir.mkdir(parents=True, exist_ok=True)
    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=5)
    on_disk = existing_papers(data_dir)

    rows: dict[str, dict] = {}
    new = 0
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    for result in client.results(search):
        bid = base_id(result.get_short_id())
        if bid in on_disk:
            rows[bid] = to_row(result, on_disk[bid].name)
            continue
        if metadata_only:  # only describe what is already on disk
            continue
        filename = safe_filename(result.get_short_id(), result.title)
        try:
            download(pdf_url(result), data_dir / filename)
        except Exception as exc:  # network hiccup or withdrawn paper: log it, keep going
            log.warning("download failed for %s: %s", result.get_short_id(), exc)
            continue
        time.sleep(3)  # arXiv's rate guidance
        new += 1
        on_disk[bid] = data_dir / filename
        rows[bid] = to_row(result, filename)

    # Seeded or older PDFs that the search above did not return still get metadata.
    missing = [p for b, p in on_disk.items() if b not in rows]
    if missing:
        ids = [ID_IN_FILENAME_RE.match(p.name)["base"] + ID_IN_FILENAME_RE.match(p.name)["version"] for p in missing]
        by_base = {base_id(i): p for i, p in zip(ids, missing)}
        for result in client.results(arxiv.Search(id_list=ids)):
            bid = base_id(result.get_short_id())
            rows[bid] = to_row(result, by_base[bid].name)

    log.info("%d new PDFs downloaded, %d papers in metadata", new, len(rows))
    return sorted(rows.values(), key=lambda r: r["arxiv_id"])


def write_metadata(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--category", default="cs.AI")
    parser.add_argument("--max", type=int, default=50, help="how many of the newest papers to pull")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args(argv)
    if args.max <= 0:
        parser.error("--max must be positive")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("arxiv").setLevel(logging.WARNING)
    rows = pull(args.data, args.category, args.max, args.metadata_only)
    write_metadata(rows, args.data / "metadata.csv")
    log.info("Wrote %s", args.data / "metadata.csv")


if __name__ == "__main__":
    main()
