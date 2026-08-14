"""
Step 2: Clean raw Reddit rows into a reusable processed dataset.

Reads:  data/raw/raw_comments.csv
Writes: data/processed/processed_comments.csv
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

from src.cleaning.filters import (
    duplicate_key,
    is_bot,
    is_emoji_only,
    is_empty_or_deleted,
    is_irrelevant,
    is_spam,
    is_too_short,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "raw_comments.csv"
PROCESSED_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "processed_comments.csv"

EXPECTED_COLUMNS = [
    "source",
    "post_id",
    "comment_id",
    "text",
    "subreddit",
    "timestamp",
    "url",
    "engagement",
]


def load_raw(path: Path = RAW_INPUT_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Raw file not found: {path}. Run Reddit ingestion first:\n"
            "  python -m src.ingestion.reddit_ingest"
        )
    df = pd.read_csv(path)
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Raw CSV missing columns: {missing}")
    return df[EXPECTED_COLUMNS].copy()


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Apply sequential cleaning rules and return (cleaned_df, removal_counts).

    Counts are exclusive by stage (first matching rule wins).
    """
    working = df.copy()
    working["text"] = working["text"].fillna("").astype(str)
    start_count = len(working)
    removals: dict[str, int] = {}

    def drop_mask(mask: pd.Series, reason: str) -> None:
        nonlocal working
        removed = int(mask.sum())
        removals[reason] = removed
        working = working.loc[~mask].copy()

    drop_mask(working["text"].map(is_empty_or_deleted), "empty_or_deleted")
    drop_mask(working["text"].map(is_emoji_only), "emoji_only")
    drop_mask(working["text"].map(is_too_short), "too_short")
    drop_mask(working["text"].map(is_bot), "bot_like")
    drop_mask(working["text"].map(is_spam), "spam_or_promo")
    drop_mask(working["text"].map(is_irrelevant), "irrelevant")

    # Exact ID duplicates (same comment/post id), then near-duplicate text
    before = len(working)
    working = working.drop_duplicates(subset=["source", "post_id", "comment_id"], keep="first")
    removals["duplicate_ids"] = before - len(working)

    before = len(working)
    working = working.copy()
    working["_dup_key"] = working["text"].map(duplicate_key)
    working = working.drop_duplicates(subset=["_dup_key"], keep="first")
    working = working.drop(columns=["_dup_key"])
    removals["duplicate_text"] = before - len(working)

    removals["total_removed"] = start_count - len(working)
    removals["rows_in"] = start_count
    removals["rows_out"] = len(working)
    return working.reset_index(drop=True), removals


def save_processed(df: pd.DataFrame, path: Path = PROCESSED_OUTPUT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info("Wrote %s cleaned rows to %s", len(df), path)
    return path


def format_report(removals: dict[str, int]) -> str:
    lines = [
        "Cleaning report",
        f"  rows in:            {removals.get('rows_in', 0)}",
        f"  empty/deleted:      {removals.get('empty_or_deleted', 0)}",
        f"  emoji-only:         {removals.get('emoji_only', 0)}",
        f"  too short:          {removals.get('too_short', 0)}",
        f"  bot-like:           {removals.get('bot_like', 0)}",
        f"  spam/promo:         {removals.get('spam_or_promo', 0)}",
        f"  irrelevant:         {removals.get('irrelevant', 0)}",
        f"  duplicate ids:      {removals.get('duplicate_ids', 0)}",
        f"  duplicate text:     {removals.get('duplicate_text', 0)}",
        f"  total removed:      {removals.get('total_removed', 0)}",
        f"  rows out:           {removals.get('rows_out', 0)}",
    ]
    return "\n".join(lines)


def run(
    input_path: Path = RAW_INPUT_PATH,
    output_path: Path = PROCESSED_OUTPUT_PATH,
) -> tuple[Path, dict[str, int]]:
    raw_df = load_raw(input_path)
    cleaned_df, removals = clean_dataframe(raw_df)
    if cleaned_df.empty:
        logger.warning("All rows were removed. Check filters or raw input quality.")
    out = save_processed(cleaned_df, output_path)
    return out, removals


def main() -> None:
    try:
        output, removals = run()
        report = format_report(removals)
        logger.info("\n%s", report)
        print(report)
        print(f"\nSuccess: saved cleaned data to {output}")
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        logger.error("Cleaning failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
