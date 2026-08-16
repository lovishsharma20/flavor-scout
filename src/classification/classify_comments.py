"""
Step 3: LLM classification of cleaned Reddit comments.

Reads:  data/processed/processed_comments.csv
Writes: data/processed/analyzed_comments.csv
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from pydantic import ValidationError

from config import (
    CLASSIFY_LIMIT,
    CLASSIFY_MAX_RETRIES,
    CLASSIFY_MODEL,
    CLASSIFY_SLEEP_SECONDS,
)
from src.classification.schemas import CommentAnalysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "processed_comments.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "analyzed_comments.csv"

SYSTEM_PROMPT = """
You are a data analyst classifying Reddit posts/comments about protein powders,
whey, supplements, electrolytes, and flavor preferences.

Return ONLY valid JSON matching this schema:
{
  "relevant": boolean,
  "flavor": string or null,
  "category": "preference" | "complaint" | "request" | "recommendation" | "comparison" | "other",
  "sentiment": "positive" | "negative" | "neutral" | "mixed",
  "intent": "share_preference" | "complain" | "request_flavor" | "recommend" | "compare" | "ask_question" | "other",
  "purchase_intent": "none" | "low" | "medium" | "high",
  "pain_point": string or null,
  "brand_fit": "strong" | "moderate" | "weak" | "unknown",
  "confidence": number between 0 and 1
}

Rules:
- Focus on taste/flavor signals when present.
- If no clear flavor is named, set flavor to null.
- If no clear pain point, set pain_point to null.
- Be conservative with purchase_intent and brand_fit.
""".strip()


def load_openai_client() -> OpenAI:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.startswith("your_"):
        raise ValueError(
            "Missing OPENAI_API_KEY in .env. "
            "Copy .env.example to .env and set your OpenAI API key."
        )
    return OpenAI(api_key=api_key)


def load_comments(path: Path = INPUT_PATH, limit: int = CLASSIFY_LIMIT) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Processed file not found: {path}. Run cleaning first:\n"
            "  python -m src.cleaning.clean_comments"
        )
    df = pd.read_csv(path)
    if "text" not in df.columns:
        raise ValueError("Input CSV must include a 'text' column.")
    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.strip().astype(bool)]
    if df.empty:
        raise ValueError("No non-empty comments found in processed CSV.")
    return df.head(limit).reset_index(drop=True)


def _extract_json_payload(content: str) -> dict[str, Any]:
    """Parse model text into a JSON object (handles accidental fences)."""
    text = (content or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


def classify_text(client: OpenAI, text: str) -> CommentAnalysis:
    """Call OpenAI and validate the response with Pydantic."""
    last_error: Exception | None = None

    for attempt in range(1, CLASSIFY_MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=CLASSIFY_MODEL,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Classify this Reddit text:\n\n{text}",
                    },
                ],
            )
            content = response.choices[0].message.content or ""
            payload = _extract_json_payload(content)
            return CommentAnalysis.model_validate(payload)

        except (RateLimitError, APITimeoutError, APIError) as exc:
            last_error = exc
            wait = min(2 ** attempt, 20)
            logger.warning(
                "OpenAI API error on attempt %s/%s: %s. Retrying in %ss.",
                attempt,
                CLASSIFY_MAX_RETRIES,
                exc,
                wait,
            )
            time.sleep(wait)
        except (json.JSONDecodeError, ValidationError, TypeError, KeyError, IndexError) as exc:
            last_error = exc
            logger.warning(
                "Invalid AI response on attempt %s/%s: %s",
                attempt,
                CLASSIFY_MAX_RETRIES,
                exc,
            )
            time.sleep(1)

    raise RuntimeError(f"Failed to classify text after retries: {last_error}")


def analyze_dataframe(df: pd.DataFrame, client: OpenAI) -> tuple[pd.DataFrame, int, int]:
    """
    Classify each row. Returns (result_df, success_count, fail_count).
    Failed rows are skipped (not written).
    """
    rows: list[dict[str, Any]] = []
    success = 0
    failed = 0

    total = len(df)
    for idx, record in df.iterrows():
        text = str(record["text"]).strip()
        logger.info("Classifying %s/%s ...", idx + 1, total)
        try:
            analysis = classify_text(client, text)
            row = record.to_dict()
            row.update(analysis.model_dump())
            rows.append(row)
            success += 1
        except Exception as exc:
            failed += 1
            logger.error("Skipping row %s due to error: %s", idx, exc)

        if CLASSIFY_SLEEP_SECONDS > 0 and idx < total - 1:
            time.sleep(CLASSIFY_SLEEP_SECONDS)

    if not rows:
        raise RuntimeError("No comments were successfully analyzed.")

    result = pd.DataFrame(rows)
    return result, success, failed


def save_analyzed(df: pd.DataFrame, path: Path = OUTPUT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info("Wrote %s analyzed rows to %s", len(df), path)
    return path


def run(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
    limit: int = CLASSIFY_LIMIT,
) -> tuple[Path, int, int]:
    client = load_openai_client()
    comments = load_comments(input_path, limit=limit)
    logger.info("Loaded %s comments for classification (limit=%s)", len(comments), limit)
    analyzed, success, failed = analyze_dataframe(comments, client)
    out = save_analyzed(analyzed, output_path)
    return out, success, failed


def main() -> None:
    try:
        output, success, failed = run()
        print(f"Successfully analyzed: {success}")
        print(f"Failed / skipped:      {failed}")
        print(f"Saved to:              {output}")
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("Classification failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
