"""Parse the arXiv PDFs in data/ into cleaned, deduplicated chunks.

    python src/parse_and_chunk.py
    python src/parse_and_chunk.py --input data --output data/processed --max-words 120

Writes two files to the output folder:
    chunks.jsonl      one JSON object per chunk, ready for embedding
    parse_report.csv  one row per paper, so parse problems show up in one place

Cleaning steps, in order:
    1. extract text per page with pypdf
    2. drop running headers/footers, page numbers and the arXiv side stamp
    3. cut acknowledgments and the reference list, keep appendices
    4. split into sentences and pack them into chunks with sentence overlap
    5. drop chunks where pypdf lost the spaces between words (usually figure text)
    6. drop exact and near-duplicate chunks (MinHash + LSH, verified with Jaccard)

Only needs pypdf.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import logging
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

log = logging.getLogger("parse_and_chunk")

FILENAME_RE = re.compile(r"^(?P<arxiv_id>\d{4}\.\d{4,5}v\d+)_(?P<title>.+)$")
ARXIV_STAMP_RE = re.compile(r"arXiv:\d{4}\.\d{4,5}v\d+\s+\[[\w.\-]+\]\s+\d{1,2}\s+\w{3}\s+\d{4}")
PAGE_NUMBER_RE = re.compile(r"^\s*(page\s+)?\d{1,3}(\s+of\s+\d{1,3})?\s*$", re.IGNORECASE)
REFERENCES_RE = re.compile(r"^\s*(\d{1,2}\.?\s+)?(references|bibliography)\s*$", re.IGNORECASE)
ACKNOWLEDGMENTS_RE = re.compile(r"^\s*(\d{1,2}\.?\s+)?acknowledge?ments?\s*$", re.IGNORECASE)
# First appendix heading after the references: "Appendix ...", "A Title" or "A Title 19" (TOC line).
APPENDIX_RE = re.compile(r"^\s*(appendix\b.{0,80}|A\.?\s+[A-Z][^\d]{2,70}(\s+\d{1,3})?)\s*$", re.IGNORECASE)
SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'])")
WORD_RE = re.compile(r"[a-z0-9]+")

LOW_QUALITY_SHARE = 0.25  # flag a paper when more than this share of its chunks came out garbled


@dataclass
class Chunk:
    chunk_id: str
    arxiv_id: str
    title: str
    source_file: str
    section: str  # "body" or "appendix"
    page_start: int
    page_end: int
    chunk_index: int
    n_words: int
    text: str


@dataclass
class PaperReport:
    arxiv_id: str
    source_file: str
    pages: int = 0
    words_extracted: int = 0
    words_kept: int = 0  # after header/footer/reference cleanup, before chunk overlap
    references_removed: bool = False
    appendix_kept: bool = False
    chunks: int = 0
    dropped_garbled: int = 0
    dropped_duplicates: int = 0
    status: str = "ok"


# ---------------------------------------------------------------- extraction


def extract_pages(pdf_path: Path) -> list[str]:
    reader = PdfReader(pdf_path)
    return [page.extract_text() or "" for page in reader.pages]


def _line_key(line: str) -> str:
    """Normalize a line so the same header matches across pages (page numbers differ)."""
    return re.sub(r"\d+", "#", line.strip().lower())


def strip_running_lines(pages: list[str], edge: int = 2, min_share: float = 0.3) -> list[list[str]]:
    """Return each page as a list of lines, minus headers, footers and page numbers.

    A line counts as a running header/footer if it is short, sits in the first or
    last `edge` lines of a page, and the same (digit-normalized) line shows up there
    on at least `min_share` of the pages.
    """
    page_lines = [[ln.strip() for ln in page.splitlines() if ln.strip()] for page in pages]

    edge_counts: Counter[str] = Counter()
    for lines in page_lines:
        if len(lines) > 2 * edge:
            edge_counts.update({_line_key(ln) for ln in lines[:edge] + lines[-edge:] if len(ln.split()) <= 15})
    threshold = max(3, int(min_share * len(pages)))
    running = {key for key, count in edge_counts.items() if count >= threshold}

    cleaned = []
    for lines in page_lines:
        n = len(lines)
        kept = []
        for i, ln in enumerate(lines):
            at_edge = n > 2 * edge and (i < edge or i >= n - edge)
            if at_edge and (_line_key(ln) in running or PAGE_NUMBER_RE.match(ln)):
                continue
            ln = ARXIV_STAMP_RE.sub("", ln).strip()
            if ln:
                kept.append(ln)
        cleaned.append(kept)
    return cleaned


def remove_back_matter(page_lines: list[list[str]]) -> tuple[list[tuple[int, str, str]], bool, bool]:
    """Flatten to (page_no, section, line) and cut acknowledgments + references.

    Anything after the reference list that starts with an appendix heading is kept
    and tagged "appendix", since appendices often hold the implementation details
    an analyst would ask about.
    """
    flat = [(p + 1, ln) for p, lines in enumerate(page_lines) for ln in lines]
    # Ignore matches in the first 30% of the paper (tables of contents, related-work mentions).
    start_search = int(len(flat) * 0.3)

    ref_idx = next((i for i in range(start_search, len(flat)) if REFERENCES_RE.match(flat[i][1])), None)
    if ref_idx is None:
        return [(p, "body", ln) for p, ln in flat], False, False

    cut_idx = ref_idx
    for i in range(max(start_search, ref_idx - 80), ref_idx):
        if ACKNOWLEDGMENTS_RE.match(flat[i][1]):
            cut_idx = i
            break

    app_idx = next((i for i in range(ref_idx + 1, len(flat)) if APPENDIX_RE.match(flat[i][1])), None)

    out = [(p, "body", ln) for p, ln in flat[:cut_idx]]
    if app_idx is not None:
        out += [(p, "appendix", ln) for p, ln in flat[app_idx:]]
    return out, True, app_idx is not None


def join_lines(lines: list[str]) -> str:
    """Join PDF lines into running text and repair words hyphenated across lines."""
    text = ""
    for ln in lines:
        if text.endswith("-") and ln[:1].islower():
            text = text[:-1] + ln
        else:
            text = f"{text} {ln}" if text else ln
    text = text.replace("\ufb01", "fi").replace("\ufb02", "fl").replace("\u00ad", "")  # ligatures, soft hyphen
    return re.sub(r"\s+", " ", text).strip()


# ------------------------------------------------------------------ chunking


def split_sentences(lines: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Split page-tagged lines into (page_no, sentence) pairs."""
    pieces, offsets, pages = [], [], []
    pos = 0
    for page_no, group in _group_by_page(lines):
        text = join_lines(group)
        if not text:
            continue
        offsets.append(pos)
        pages.append(page_no)
        pieces.append(text)
        pos += len(text) + 1
    full = " ".join(pieces)

    sentences = []
    start = 0
    for match in SENTENCE_END_RE.finditer(full):
        sentences.append((start, full[start : match.start()]))
        start = match.end()
    sentences.append((start, full[start:]))

    return [(pages[bisect.bisect_right(offsets, s) - 1], sent.strip()) for s, sent in sentences if sent.strip()]


def _group_by_page(lines: list[tuple[int, str]]):
    group, current = [], None
    for page_no, ln in lines:
        if page_no != current and group:
            yield current, group
            group = []
        current = page_no
        group.append(ln)
    if group:
        yield current, group


def pack_chunks(sentences: list[tuple[int, str]], max_words: int, overlap_words: int) -> list[tuple[int, int, str]]:
    """Greedily pack sentences into chunks of at most `max_words` words.

    Consecutive chunks share whole sentences worth about `overlap_words` words, so
    an idea split across a boundary still lands intact in one of them. Sentences
    longer than `max_words` (tables, run-on extraction) are hard-split by words.
    """
    units: list[tuple[int, list[str]]] = []
    for page_no, sent in sentences:
        words = sent.split()
        for i in range(0, len(words), max_words):
            units.append((page_no, words[i : i + max_words]))

    chunks = []
    i = 0
    while i < len(units):
        j, n_words = i, 0
        while j < len(units) and n_words + len(units[j][1]) <= max_words:
            n_words += len(units[j][1])
            j += 1
        if j == i:  # single unit already at the limit
            j = i + 1
        text = " ".join(" ".join(w) for _, w in units[i:j])
        chunks.append((units[i][0], units[j - 1][0], text))
        if j >= len(units):
            break

        # Step back over whole sentences to build the overlap, but always move forward.
        back, carried = j, 0
        while back - 1 > i and carried + len(units[back - 1][1]) <= overlap_words:
            back -= 1
            carried += len(units[back][1])
        i = back
    return chunks


def is_garbled(text: str, max_token_len: int = 25, max_share: float = 0.2) -> bool:
    """True when pypdf dropped the spaces between words, e.g. text inside figures."""
    tokens = text.split()
    if not tokens:
        return True
    long_chars = sum(len(t) for t in tokens if len(t) >= max_token_len)
    return long_chars / max(1, sum(len(t) for t in tokens)) > max_share


# --------------------------------------------------------------- dedup


class NearDuplicateFilter:
    """Streaming near-duplicate filter: MinHash signatures bucketed with LSH.

    Candidates that share a bucket are confirmed with exact Jaccard similarity on
    word 5-gram shingles, so the LSH only decides what gets compared, not what
    gets dropped. The first chunk seen is the one kept.
    """

    _PRIME = (1 << 61) - 1

    def __init__(self, threshold: float = 0.85, num_perm: int = 128, bands: int = 32, shingle: int = 5, seed: int = 7):
        assert num_perm % bands == 0
        self.threshold = threshold
        self.bands = bands
        self.rows = num_perm // bands
        self.shingle = shingle
        rng = random.Random(seed)
        self.perms = [(rng.randrange(1, self._PRIME), rng.randrange(0, self._PRIME)) for _ in range(num_perm)]
        self.buckets: dict[tuple, list[int]] = defaultdict(list)
        self.kept_shingles: list[set[int]] = []
        self.exact: set[str] = set()

    def _shingles(self, text: str) -> set[int]:
        words = WORD_RE.findall(text.lower())
        k = self.shingle
        grams = [" ".join(words[i : i + k]) for i in range(max(1, len(words) - k + 1))]
        return {int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "big") for g in grams}

    def _signature(self, shingles: set[int]) -> list[int]:
        return [min((a * s + b) % self._PRIME for s in shingles) for a, b in self.perms]

    def is_duplicate(self, text: str) -> bool:
        norm = " ".join(WORD_RE.findall(text.lower()))
        digest = hashlib.sha1(norm.encode()).hexdigest()
        if digest in self.exact:
            return True

        shingles = self._shingles(text)
        sig = self._signature(shingles)
        keys = [(b, tuple(sig[b * self.rows : (b + 1) * self.rows])) for b in range(self.bands)]

        candidates = {idx for key in keys for idx in self.buckets.get(key, ())}
        for idx in candidates:
            other = self.kept_shingles[idx]
            if len(shingles & other) / len(shingles | other) >= self.threshold:
                return True

        idx = len(self.kept_shingles)
        self.kept_shingles.append(shingles)
        self.exact.add(digest)
        for key in keys:
            self.buckets[key].append(idx)
        return False


# ------------------------------------------------------------------ pipeline


def process_paper(
    pdf_path: Path, max_words: int, overlap_words: int, dedup: NearDuplicateFilter
) -> tuple[list[Chunk], PaperReport]:
    match = FILENAME_RE.match(pdf_path.stem)
    arxiv_id = match["arxiv_id"] if match else pdf_path.stem
    title = match["title"] if match else pdf_path.stem
    report = PaperReport(arxiv_id=arxiv_id, source_file=pdf_path.name)

    try:
        pages = extract_pages(pdf_path)
    except (PdfReadError, OSError, ValueError) as exc:  # corrupt or encrypted PDF: record it, keep going
        report.status = f"extract_failed: {exc.__class__.__name__}"
        return [], report

    report.pages = len(pages)
    report.words_extracted = sum(len(p.split()) for p in pages)
    if report.words_extracted == 0:
        report.status = "no_text (scanned PDF?)"
        return [], report

    lines, report.references_removed, report.appendix_kept = remove_back_matter(strip_running_lines(pages))

    chunks: list[Chunk] = []
    for section in ("body", "appendix"):
        sentences = split_sentences([(p, ln) for p, s, ln in lines if s == section])
        report.words_kept += sum(len(sent.split()) for _, sent in sentences)
        for page_start, page_end, text in pack_chunks(sentences, max_words, overlap_words):
            if is_garbled(text):
                report.dropped_garbled += 1
                continue
            if dedup.is_duplicate(text):
                report.dropped_duplicates += 1
                continue
            idx = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{arxiv_id}::{idx:04d}",
                    arxiv_id=arxiv_id,
                    title=title,
                    source_file=pdf_path.name,
                    section=section,
                    page_start=page_start,
                    page_end=page_end,
                    chunk_index=idx,
                    n_words=len(text.split()),
                    text=text,
                )
            )

    report.chunks = len(chunks)
    produced = report.chunks + report.dropped_garbled + report.dropped_duplicates
    if produced and report.dropped_garbled / produced > LOW_QUALITY_SHARE:
        report.status = "warn: pypdf lost word spacing, most chunks dropped"
    elif not report.references_removed:
        report.status = "warn: no references heading found"
    return chunks, report


def run(
    input_dir: Path, output_dir: Path, max_words: int, overlap_words: int, dedup_threshold: float
) -> list[PaperReport]:
    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    dedup = NearDuplicateFilter(threshold=dedup_threshold)
    reports = []

    with open(output_dir / "chunks.jsonl", "w", encoding="utf-8") as out:
        for pdf_path in pdfs:
            chunks, report = process_paper(pdf_path, max_words, overlap_words, dedup)
            out.writelines(json.dumps(asdict(chunk), ensure_ascii=False) + "\n" for chunk in chunks)
            reports.append(report)
            log.info("%s: %d chunks (%s)", report.arxiv_id, report.chunks, report.status)

    with open(output_dir / "parse_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(reports[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(r) for r in reports)

    return reports


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--max-words",
        type=int,
        default=120,
        help="words per chunk, sized to fit all-MiniLM-L6-v2's 256-token input (see Issue #4)",
    )
    parser.add_argument("--overlap-words", type=int, default=20)
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.85,
        help="Jaccard similarity on word 5-grams above which a chunk is dropped",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("pypdf").setLevel(logging.ERROR)  # font-encoding warnings, not actionable here
    reports = run(args.input, args.output, args.max_words, args.overlap_words, args.dedup_threshold)

    total = sum(r.chunks for r in reports)
    log.info("Done: %d papers, %d chunks -> %s", len(reports), total, args.output)
    for r in reports:
        if r.status != "ok":
            log.warning("%s: %s", r.arxiv_id, r.status)
    if any(r.chunks == 0 for r in reports):
        sys.exit(1)


if __name__ == "__main__":
    main()
