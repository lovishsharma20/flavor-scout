"""
Step 1: Stream Amazon Reviews 2023 Sports_and_Outdoors reviews.

Official source (historical, not live):
  https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Sports_and_Outdoors.jsonl.gz
  Docs: https://amazon-reviews-2023.github.io/

Streams the remote .jsonl.gz (does NOT save the ~986 MB file locally).
Filters Flavor Scout–relevant reviews and writes data/raw/amazon_reviews.csv.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.request import Request, urlopen

import pandas as pd

from config import (
    AMAZON_CATEGORY,
    AMAZON_MIN_REVIEW_CHARS,
    AMAZON_REVIEW_KEYWORDS,
    AMAZON_REVIEW_URL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "amazon_reviews.csv"

OUTPUT_COLUMNS = [
    "source",
    "category",
    "product_id",
    "parent_asin",
    "review_title",
    "review_text",
    "rating",
    "timestamp",
    "helpful_votes",
    "verified_purchase",
]


def stream_review_records(
    url: str = AMAZON_REVIEW_URL,
    max_records: int | None = None,
) -> Iterator[dict[str, Any]]:
    """
    Yield review JSON objects from the remote .jsonl.gz without writing it to disk.

    Uses a streaming HTTP response + gzip decompressor so only a buffer is held
    in memory (not the full compressed file).
    """
    request = Request(
        url,
        headers={"User-Agent": "FlavorScout/0.1 (research MVP; streaming reader)"},
    )
    logger.info("Opening remote stream: %s", url)
    with urlopen(request, timeout=120) as response:  # noqa: S310 - fixed official URL
        with gzip.GzipFile(fileobj=response) as gz:
            # Text wrapper over binary gzip stream
            for idx, raw_line in enumerate(gz, start=1):
                if max_records is not None and idx > max_records:
                    break
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSON on source row %s", idx)
                    continue


def _to_iso_timestamp(raw_ts: Any) -> str:
    if raw_ts is None or raw_ts == "":
        return ""
    try:
        value = int(raw_ts)
    except (TypeError, ValueError):
        return str(raw_ts)
    if value > 1_000_000_000_000:  # milliseconds
        value = value / 1000.0
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def _helpful_votes(review: dict[str, Any]) -> int:
    raw = review.get("helpful_vote", review.get("helpful_votes", 0))
    try:
        return int(raw or 0)
    except (TypeError, ValueError):
        return 0


def _review_timestamp(review: dict[str, Any]) -> Any:
    return review.get("timestamp", review.get("sort_timestamp"))


def is_relevant_review(review: dict[str, Any], keywords: list[str]) -> bool:
    title = (review.get("title") or "").strip()
    text = (review.get("text") or "").strip()
    if len(text) < AMAZON_MIN_REVIEW_CHARS:
        return False
    blob = f"{title} {text}".lower()
    return any(keyword.lower() in blob for keyword in keywords)


def review_to_row(review: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "amazon_reviews_2023",
        "category": AMAZON_CATEGORY,
        "product_id": (review.get("asin") or "").strip(),
        "parent_asin": (review.get("parent_asin") or "").strip(),
        "review_title": (review.get("title") or "").strip(),
        "review_text": (review.get("text") or "").strip(),
        "rating": review.get("rating"),
        "timestamp": _to_iso_timestamp(_review_timestamp(review)),
        "helpful_votes": _helpful_votes(review),
        "verified_purchase": bool(review.get("verified_purchase", False)),
    }


def duplicate_key(row: dict[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("product_id", "")),
            str(row.get("parent_asin", "")),
            str(row.get("timestamp", "")),
            str(row.get("rating", "")),
            (row.get("review_text") or "").strip().lower(),
        ]
    )


def extract_relevant_reviews(
    max_records: int | None = None,
    keywords: list[str] | None = None,
) -> tuple[pd.DataFrame, int, int]:
    """
    Stream source reviews, keep keyword matches, drop duplicates.

    Returns (dataframe, scanned_count, extracted_before_dedupe_info via df len after dedupe).
    Also returns scanned and relevant-before-dedupe via logging; tuple is
    (df, scanned, relevant_kept_after_dedupe) — we also track relevant pre-dedupe.
    """
    keywords = keywords or AMAZON_REVIEW_KEYWORDS
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    scanned = 0
    relevant_raw = 0

    for review in stream_review_records(max_records=max_records):
        scanned += 1
        if scanned % 100_000 == 0:
            logger.info(
                "Scanned %s source records; relevant kept so far=%s",
                scanned,
                len(rows),
            )

        if not is_relevant_review(review, keywords):
            continue

        relevant_raw += 1
        row = review_to_row(review)
        key = duplicate_key(row)
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    logger.info(
        "Scan finished: scanned=%s, relevant_matched=%s, after_dedupe=%s",
        scanned,
        relevant_raw,
        len(df),
    )
    return df, scanned, relevant_raw


def probe_schema(sample_size: int = 1000) -> dict[str, Any]:
    """Read the first N source records and report field presence (no CSV write)."""
    expected_fields = [
        "rating",
        "title",
        "text",
        "asin",
        "parent_asin",
        "user_id",
        "timestamp",
        "verified_purchase",
        "helpful_vote",
    ]
    alt_fields = ["helpful_votes", "sort_timestamp"]

    records: list[dict[str, Any]] = []
    for review in stream_review_records(max_records=sample_size):
        records.append(review)

    if not records:
        raise RuntimeError("Probe failed: no records received from remote stream.")

    field_counts: dict[str, int] = {}
    all_keys: set[str] = set()
    for rec in records:
        all_keys.update(rec.keys())
        for key in rec:
            field_counts[key] = field_counts.get(key, 0) + 1

    present_expected = [f for f in expected_fields if field_counts.get(f, 0) > 0]
    missing_expected = [f for f in expected_fields if field_counts.get(f, 0) == 0]
    present_alts = [f for f in alt_fields if field_counts.get(f, 0) > 0]

    relevant = sum(1 for r in records if is_relevant_review(r, AMAZON_REVIEW_KEYWORDS))

    report = {
        "records_read": len(records),
        "unique_keys": sorted(all_keys),
        "field_counts": dict(sorted(field_counts.items())),
        "expected_present": present_expected,
        "expected_missing": missing_expected,
        "alt_fields_present": present_alts,
        "relevant_in_sample": relevant,
        "sample_record_keys": sorted(records[0].keys()),
        "sample_preview": {
            "rating": records[0].get("rating"),
            "asin": records[0].get("asin"),
            "parent_asin": records[0].get("parent_asin"),
            "title": (records[0].get("title") or "")[:80],
            "text": (records[0].get("text") or "")[:120],
            "timestamp": records[0].get("timestamp", records[0].get("sort_timestamp")),
            "helpful_vote": records[0].get("helpful_vote", records[0].get("helpful_votes")),
            "verified_purchase": records[0].get("verified_purchase"),
        },
    }
    return report


def save_reviews(df: pd.DataFrame, path: Path = RAW_OUTPUT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info("Wrote %s rows to %s", len(df), path)
    return path


def run_extract(max_records: int | None = None) -> tuple[Path, int, int, int]:
    df, scanned, relevant_raw = extract_relevant_reviews(max_records=max_records)
    if df.empty:
        raise RuntimeError("No relevant reviews extracted. Check keywords / connection.")
    out = save_reviews(df)
    return out, scanned, relevant_raw, len(df)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream Amazon Reviews 2023 Sports_and_Outdoors for Flavor Scout."
    )
    parser.add_argument(
        "--probe",
        type=int,
        nargs="?",
        const=1000,
        default=None,
        help="Connection/schema test only: read first N records (default 1000). No CSV.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap on source records scanned (for partial runs).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Scan the full remote stream (can take a long time).",
    )
    args = parser.parse_args()

    try:
        if args.probe is not None:
            report = probe_schema(sample_size=args.probe)
            print("=== Amazon stream probe (no full scan) ===")
            print(f"Records read:           {report['records_read']}")
            print(f"Keys in sample:         {', '.join(report['sample_record_keys'])}")
            print(f"Expected fields found: {', '.join(report['expected_present']) or '(none)'}")
            print(f"Expected fields missing:{', '.join(report['expected_missing']) or '(none)'}")
            print(f"Alt fields present:    {', '.join(report['alt_fields_present']) or '(none)'}")
            print(f"Relevant in sample:     {report['relevant_in_sample']}")
            print("Sample preview:")
            for key, value in report["sample_preview"].items():
                print(f"  {key}: {value}")
            ok = (
                report["records_read"] > 0
                and "text" in report["sample_record_keys"]
                and "rating" in report["sample_record_keys"]
                and (
                    "asin" in report["sample_record_keys"]
                    or "parent_asin" in report["sample_record_keys"]
                )
            )
            print(f"Structure OK:            {ok}")
            sys.exit(0 if ok else 1)

        if not args.full and args.max_records is None:
            print(
                "Refusing to start a full scan without --full.\n"
                "First run a probe:\n"
                "  python -m src.ingestion.amazon_ingest --probe 1000\n"
                "When approved, run:\n"
                "  python -m src.ingestion.amazon_ingest --full"
            )
            sys.exit(2)

        max_records = None if args.full else args.max_records
        out, scanned, relevant_raw, kept = run_extract(max_records=max_records)
        print("Amazon review extraction complete (historical sample).")
        print(f"Source records scanned:   {scanned}")
        print(f"Relevant matched:         {relevant_raw}")
        print(f"After dedupe (saved):     {kept}")
        print(f"Saved to:                 {out}")
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        logger.error("Amazon ingestion failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
