"""Bulk-load Elsevier OA articles into SQLite + ChromaDB.

Usage (from CapstoneProject/ root with the venv active):

    python scripts/seed_elsevier.py                         # 50 random articles
    python scripts/seed_elsevier.py --n 10000               # 10k random articles
    python scripts/seed_elsevier.py --n 10000 --stratified  # 10k spanning all fields
    python scripts/seed_elsevier.py --n 0                   # ALL articles
    python scripts/seed_elsevier.py --reset --n 10000 --stratified  # wipe + reload

The script calls the service layer directly — the FastAPI server does NOT need
to be running.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

# ── Allow running from the scripts/ directory ────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.db.session import SessionLocal, init_db
from app.models.database import Base, Document
from app.services.document_processor import chunk_text, extract_elsevier_metadata, extract_text
from app.services.vector_store import add_chunks, delete_document_chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ARCHIVE_DIR = Path("/home/jalzate/Downloads/archive/json/json")


# ── helpers ───────────────────────────────────────────────────────────────────

def _reset_data() -> None:
    """Drop and recreate all SQLite tables and wipe ChromaDB collection."""
    from sqlalchemy import create_engine
    import chromadb

    log.warning("Resetting database and ChromaDB …")
    engine = create_engine(settings.database_url)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    try:
        client.delete_collection(settings.chroma_collection_name)
        log.info("ChromaDB collection '%s' deleted.", settings.chroma_collection_name)
    except Exception:
        pass  # collection did not exist yet


def _already_loaded(db, path: Path) -> bool:
    return db.query(Document).filter(Document.file_path == str(path)).first() is not None


def _load_article(db, path: Path) -> int:
    """Extract, chunk, embed, and record one article. Returns chunk count."""
    text = extract_text(str(path), "json")
    chunks = chunk_text(text)

    # Extract rich metadata (authors, year, doi, subjareas) for ChromaDB
    try:
        doc_metadata = extract_elsevier_metadata(str(path))
    except Exception:
        doc_metadata = None

    doc = Document(
        filename=path.name,
        file_path=str(path),
        file_type="json",
        status="processed",
        chunk_count=len(chunks),
    )
    db.add(doc)
    db.flush()  # populate doc.id before passing to ChromaDB

    add_chunks(doc_id=doc.id, filename=path.name, chunks=chunks, doc_metadata=doc_metadata)
    db.commit()
    return len(chunks)


# ── stratified sampler ────────────────────────────────────────────────────────

def _stratified_sample(all_files: list[Path], n: int) -> list[Path]:
    """Sample n files with proportional coverage across Elsevier subject areas.

    Scans every JSON file for its primary subject area code, then draws
    ceil(n / num_areas) files from each area.
    This adds a one-time scan overhead (~30-60 s for 40 k files) but ensures
    the resulting knowledge base spans all research domains evenly.
    """
    import json

    log.info("Scanning archive for subject area distribution (%d files)…", len(all_files))
    buckets: dict[str, list[Path]] = defaultdict(list)

    for i, f in enumerate(all_files, 1):
        if i % 5000 == 0:
            log.info("  scanned %d / %d …", i, len(all_files))
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            areas = data.get("metadata", {}).get("subjareas", [])
            primary = areas[0] if areas else "UNCL"
        except Exception:
            primary = "UNCL"
        buckets[primary].append(f)

    area_codes = sorted(buckets)
    log.info(
        "Found %d subject areas: %s%s",
        len(area_codes),
        ", ".join(area_codes[:8]),
        "…" if len(area_codes) > 8 else "",
    )

    per_area = max(1, math.ceil(n / len(area_codes)))
    selected: list[Path] = []
    for area in area_codes:
        files = buckets[area]
        selected.extend(random.sample(files, min(per_area, len(files))))

    random.shuffle(selected)
    return selected[:n]


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Seed Elsevier articles into the knowledge base.")
    parser.add_argument("--n", type=int, default=50, help="Number of articles to load (0 = all)")
    parser.add_argument("--reset", action="store_true", help="Wipe existing data before seeding")
    parser.add_argument(
        "--stratified",
        action="store_true",
        help="Sample proportionally across subject areas instead of randomly",
    )
    args = parser.parse_args()

    if not ARCHIVE_DIR.exists():
        log.error("Archive directory not found: %s", ARCHIVE_DIR)
        sys.exit(1)

    if args.reset:
        _reset_data()

    init_db()

    all_files = sorted(ARCHIVE_DIR.glob("*.json"))
    if not all_files:
        log.error("No JSON files found in %s", ARCHIVE_DIR)
        sys.exit(1)

    target = all_files if args.n == 0 else (
        _stratified_sample(all_files, min(args.n, len(all_files)))
        if args.stratified
        else random.sample(all_files, min(args.n, len(all_files)))
    )
    log.info(
        "Found %d articles in archive. Loading %d (%s)…",
        len(all_files),
        len(target),
        "stratified" if args.stratified else "random",
    )

    loaded = skipped = failed = 0

    db = SessionLocal()
    try:
        for i, path in enumerate(target, 1):
            if _already_loaded(db, path):
                log.debug("Skipping (already in DB): %s", path.name)
                skipped += 1
                continue
            try:
                chunk_count = _load_article(db, path)
                log.info("[%d/%d] ✓ %s  → %d chunks", i, len(target), path.name, chunk_count)
                loaded += 1
            except Exception as exc:
                log.warning("[%d/%d] ✗ %s  — %s", i, len(target), path.name, exc)
                db.rollback()
                failed += 1
    finally:
        db.close()

    print(
        f"\nDone — loaded: {loaded} | skipped (already exist): {skipped} | failed: {failed}"
    )


if __name__ == "__main__":
    main()
