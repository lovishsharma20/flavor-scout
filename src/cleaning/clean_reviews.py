"""
Step 2: Enrich Amazon reviews with product metadata, then rule-based cleaning.

Reads:  data/raw/amazon_reviews.csv  (unchanged)
Writes: data/processed/processed_reviews.csv

Does NOT use an LLM. Relevance classification belongs to Step 3.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

from config import AMAZON_REVIEW_KEYWORDS
from src.cleaning.filters import (
    duplicate_key,
    is_bot,
    is_emoji_only,
    is_empty_or_deleted,
    is_too_short,
    normalize_text,
)
from src.ingestion.amazon_metadata import fetch_metadata_for_asins

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "amazon_reviews.csv"
PROCESSED_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "processed_reviews.csv"

SPAM_PATTERNS = [
    r"\buse\s+(my\s+)?code\b",
    r"\bpromo\s*code\b",
    r"\bdiscount\s*code\b",
    r"\baffiliate\b",
    r"\bbuy\s+now\b",
    r"\border\s+now\b",
    r"\bclick\s+(here|the\s+link)\b",
    r"\bcheck\s+out\s+my\b",
    r"\bdm\s+me\b",
    r"\blink\s+in\s+(bio|profile)\b",
    r"\bsubscribe\s+to\s+(my|our)\b",
    r"\bfollow\s+me\b",
]

OUTPUT_COLUMNS = [
    "source",
    "category",
    "product_id",
    "parent_asin",
    "product_title",
    "brand",
    "store",
    "main_category",
    "product_categories",
    "product_description",
    "average_rating",
    "rating_number",
    "price",
    "review_title",
    "review_text",
    "rating",
    "timestamp",
    "helpful_votes",
    "verified_purchase",
    "metadata_matched",
]


def load_raw(path: Path = RAW_INPUT_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Raw file not found: {path}. Run Amazon ingestion first:\n"
            "  python -m src.ingestion.amazon_ingest --full"
        )
    df = pd.read_csv(path)
    required = ["review_text", "product_id", "parent_asin"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Raw CSV missing columns: {missing}")
    return df.copy()


def enrich_with_metadata(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    wanted = df["parent_asin"].dropna().astype(str).str.strip()
    wanted = wanted[wanted.ne("") & wanted.ne("nan")].unique().tolist()
    meta_map, meta_stats = fetch_metadata_for_asins(wanted)

    meta_df = pd.DataFrame(meta_map.values())
    working = df.copy()
    drop_cols = [c for c in working.columns if c in {
        "product_title", "brand", "store", "main_category",
        "product_categories", "product_description", "average_rating",
        "rating_number", "price",
    }]
    working = working.drop(columns=drop_cols, errors="ignore")

    if meta_df.empty:
        for col in [
            "product_title", "brand", "store", "main_category",
            "product_categories", "product_description", "average_rating",
            "rating_number", "price",
        ]:
            working[col] = pd.NA
        working["metadata_matched"] = False
    else:
        working = working.merge(meta_df, on="parent_asin", how="left")
        working["metadata_matched"] = working["product_title"].notna() & (
            working["product_title"].astype(str).str.strip() != ""
        )

    matched_rows = int(working["metadata_matched"].sum())
    meta_stats["review_rows"] = len(working)
    meta_stats["review_rows_with_metadata"] = matched_rows
    meta_stats["unique_parent_asins"] = len(wanted)
    meta_stats["unique_parent_asins_matched"] = int(
        working.loc[working["metadata_matched"], "parent_asin"].nunique()
    )
    return working, meta_stats


def _is_spam_promo(text: str) -> bool:
    import re

    cleaned = normalize_text(text)
    return any(re.search(pattern, cleaned, flags=re.IGNORECASE) for pattern in SPAM_PATTERNS)


def _product_blob(row: pd.Series) -> str:
    parts = [
        row.get("product_title"),
        row.get("brand"),
        row.get("store"),
        row.get("product_categories"),
        row.get("product_description"),
        row.get("review_title"),
        row.get("review_text"),
    ]
    return " ".join("" if pd.isna(p) else str(p) for p in parts).lower()


def is_clearly_irrelevant_product(row: pd.Series) -> bool:
    """
    Drop only when the product (or review, if metadata is missing) has no
    sports-nutrition signal. Do not require 'flavor' or 'taste'.
    """
    blob = _product_blob(row)
    return not any(keyword.lower() in blob for keyword in AMAZON_REVIEW_KEYWORDS)


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    working = df.copy()
    working["review_text"] = working["review_text"].fillna("").astype(str)
    start_count = len(working)
    removals: dict[str, int] = {"rows_in": start_count}

    def drop_mask(mask: pd.Series, reason: str) -> None:
        nonlocal working
        removed = int(mask.sum())
        removals[reason] = removed
        working = working.loc[~mask].copy()

    drop_mask(working["review_text"].map(is_empty_or_deleted), "empty_or_invalid")
    drop_mask(working["review_text"].map(is_emoji_only), "unusable_noise")
    drop_mask(working["review_text"].map(is_too_short), "unusable_short")
    drop_mask(working["review_text"].map(is_bot), "bot_like")
    drop_mask(working["review_text"].map(_is_spam_promo), "spam_or_promo")
    drop_mask(working.apply(is_clearly_irrelevant_product, axis=1), "irrelevant")

    before = len(working)
    working = working.drop_duplicates(
        subset=["product_id", "parent_asin", "timestamp", "review_text"],
        keep="first",
    )
    removals["duplicate_ids"] = before - len(working)

    before = len(working)
    working = working.copy()
    working["_dup_key"] = working["review_text"].map(duplicate_key)
    working = working.drop_duplicates(subset=["_dup_key"], keep="first")
    working = working.drop(columns=["_dup_key"])
    removals["duplicate_text"] = before - len(working)

    removals["duplicates_removed"] = removals["duplicate_ids"] + removals["duplicate_text"]
    removals["noise_removed"] = (
        removals["unusable_noise"] + removals["unusable_short"] + removals["bot_like"]
    )
    removals["rows_out"] = len(working)
    removals["total_removed"] = start_count - len(working)
    return working.reset_index(drop=True), removals


def save_processed(df: pd.DataFrame, path: Path = PROCESSED_OUTPUT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [c for c in OUTPUT_COLUMNS if c in df.columns]
    extra = [c for c in df.columns if c not in ordered]
    df[ordered + extra].to_csv(path, index=False, encoding="utf-8")
    logger.info("Wrote %s cleaned rows to %s", len(df), path)
    return path


def format_report(
    removals: dict[str, int],
    meta_stats: dict[str, int],
    columns: list[str],
) -> str:
    start = removals.get("rows_in", 0)
    final = removals.get("rows_out", 0)
    retained = (100.0 * final / start) if start else 0.0
    unique_wanted = meta_stats.get("unique_parent_asins", 0)
    unique_matched = meta_stats.get("unique_parent_asins_matched", 0)
    row_matched = meta_stats.get("review_rows_with_metadata", 0)
    row_total = meta_stats.get("review_rows", start)
    asin_rate = (100.0 * unique_matched / unique_wanted) if unique_wanted else 0.0
    row_rate = (100.0 * row_matched / row_total) if row_total else 0.0
    lines = [
        "Step 2 cleaning report",
        f"  starting rows:                 {start}",
        f"  missing/invalid rows removed:  {removals.get('empty_or_invalid', 0)}",
        f"  duplicates removed:            {removals.get('duplicates_removed', 0)}",
        f"  spam/promotional removed:      {removals.get('spam_or_promo', 0)}",
        f"  unusable noise removed:        {removals.get('noise_removed', 0)}",
        f"  irrelevant records removed:    {removals.get('irrelevant', 0)}",
        f"  final rows:                    {final}",
        f"  percentage retained:           {retained:.1f}%",
        "",
        "Metadata enrichment",
        f"  unique parent_asin in reviews: {unique_wanted}",
        f"  unique parent_asin matched:    {unique_matched}",
        f"  metadata ASIN match rate:      {asin_rate:.1f}%",
        f"  review rows with title/brand:  {row_matched} / {row_total} ({row_rate:.1f}%)",
        f"  metadata rows scanned:         {meta_stats.get('meta_scanned', 0)}",
        "",
        f"Final columns: {', '.join(columns)}",
    ]
    return "\n".join(lines)


def print_samples(df: pd.DataFrame, n: int = 5) -> None:
    print("\nSample cleaned records:")
    sample_cols = [
        c for c in [
            "product_id", "parent_asin", "product_title", "brand",
            "rating", "verified_purchase", "helpful_votes", "timestamp",
            "review_title", "review_text",
        ] if c in df.columns
    ]
    preview = df[sample_cols].head(n).copy()
    if "review_text" in preview.columns:
        preview["review_text"] = preview["review_text"].astype(str).str.slice(0, 120)
    if "product_title" in preview.columns:
        preview["product_title"] = preview["product_title"].astype(str).str.slice(0, 80)
    print(preview.to_string(index=False))


def run(
    input_path: Path = RAW_INPUT_PATH,
    output_path: Path = PROCESSED_OUTPUT_PATH,
) -> tuple[Path, pd.DataFrame, dict[str, int], dict[str, int]]:
    raw_df = load_raw(input_path)
    enriched, meta_stats = enrich_with_metadata(raw_df)
    cleaned, removals = clean_dataframe(enriched)
    if cleaned.empty:
        logger.warning("All rows were removed. Check filters or raw input quality.")
    out = save_processed(cleaned, output_path)
    return out, cleaned, removals, meta_stats


def main() -> None:
    try:
        output, cleaned, removals, meta_stats = run()
        report = format_report(removals, meta_stats, list(cleaned.columns))
        print(report)
        print_samples(cleaned)
        print(f"\nOutput path: {output}")
        print("Raw dataset left unchanged at data/raw/amazon_reviews.csv")
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("Step 2 failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
