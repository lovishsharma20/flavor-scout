"""
Step 3: LLM classification of cleaned Amazon reviews (pilot).

Reads:  data/processed/processed_reviews.csv
Writes: data/processed/analyzed_reviews_pilot.csv

First run classifies CLASSIFY_PILOT_LIMIT reviews only (default 50).
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
    CLASSIFY_MAX_RETRIES,
    CLASSIFY_MODEL,
    CLASSIFY_PILOT_LIMIT,
    CLASSIFY_SLEEP_SECONDS,
)
from src.classification.schemas import ReviewAnalysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "processed_reviews.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "analyzed_reviews_pilot.csv"

ANALYSIS_COLUMNS = [
    "relevant",
    "flavor",
    "sentiment",
    "intent",
    "pain_point",
    "brand_fit",
    "reasoning",
    "confidence",
]

SYSTEM_PROMPT = """
You are Flavor Scout's classification analyst for HealthKart (MuscleBlaze, HK Vitals, TrueBasics).

Your job is to classify ONE Amazon review for flavor-discovery research.
Return ONLY valid JSON matching this schema:
{
  "relevant": boolean,
  "flavor": string or null,
  "sentiment": "positive" | "neutral" | "negative",
  "intent": "preference" | "request" | "complaint" | "purchase_intent" | "recommendation" | "other" | "none",
  "pain_point": string or null,
  "brand_fit": "strong" | "moderate" | "weak" | "none",
  "reasoning": string,
  "confidence": "high" | "medium" | "low"
}

Relevance (strict):
- relevant=true ONLY for sports nutrition / ingestible products: protein, whey, supplements,
  electrolytes/drink mixes, pre-workout, recovery drinks, nutrition gummies, vitamins sold as
  sports nutrition, similar consumables, AND the review is useful for flavor/taste/product experience.
- relevant=false for gear and unrelated items: hydration backpacks, water bladders, tote bags,
  apparel, shoes, equipment, camping gear, even if they mention "hydration".
- A relevant nutrition review may still have flavor=null if no flavor is named.

Flavor:
- Extract a specific flavor only if the review clearly mentions, requests, prefers, or criticizes one
  (e.g. chocolate, vanilla, fruit punch, mango).
- Do NOT infer a flavor from the product category, color, or brand.
- If none is named, flavor must be null.

Other rules:
- Do not fabricate pain points, brands, or flavors.
- brand_fit is about HealthKart fitness-nutrition context, not inventing a HealthKart SKU.
  Unrelated products => brand_fit "none".
- reasoning must justify the labels; do not merely restate the review.
- sentiment is about the review overall (not mixed).
- intent is the primary consumer intent; use "none" if none of the others fit.
""".strip()


def load_openai_client() -> OpenAI:
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip().strip('"').strip("'")
    if not api_key or api_key.startswith("your_"):
        raise ValueError(
            "Missing OPENAI_API_KEY in .env. "
            "Paste your OpenAI API key into .env (do not paste it into chat)."
        )
    return OpenAI(api_key=api_key)


def load_reviews(path: Path = INPUT_PATH, limit: int = CLASSIFY_PILOT_LIMIT) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Processed file not found: {path}. Run Step 2 first:\n"
            "  python -m src.cleaning.clean_reviews"
        )
    df = pd.read_csv(path)
    if "review_text" not in df.columns:
        raise ValueError("Input CSV must include a 'review_text' column.")
    df = df.copy()
    df["review_text"] = df["review_text"].fillna("").astype(str)
    df = df[df["review_text"].str.strip().astype(bool)]
    if df.empty:
        raise ValueError("No non-empty reviews found in processed CSV.")
    return df.head(limit).reset_index(drop=True)


def _extract_json_payload(content: str) -> dict[str, Any]:
    text = (content or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


def _user_payload(record: pd.Series) -> str:
    title = record.get("product_title") or ""
    brand = record.get("brand") or ""
    categories = record.get("product_categories") or ""
    review_title = record.get("review_title") or ""
    review_text = record.get("review_text") or ""
    return (
        "Classify this Amazon review for Flavor Scout.\n\n"
        f"product_title: {title}\n"
        f"brand: {brand}\n"
        f"product_categories: {categories}\n"
        f"review_title: {review_title}\n"
        f"review_text: {review_text}"
    )


def classify_review(client: OpenAI, record: pd.Series) -> ReviewAnalysis:
    last_error: Exception | None = None
    user_content = _user_payload(record)

    for attempt in range(1, CLASSIFY_MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=CLASSIFY_MODEL,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
            content = response.choices[0].message.content or ""
            payload = _extract_json_payload(content)
            return ReviewAnalysis.model_validate(payload)

        except (RateLimitError, APITimeoutError, APIError) as exc:
            last_error = exc
            message = str(exc).lower()
            if "insufficient_quota" in message:
                raise RuntimeError(
                    "OpenAI quota exceeded. Add billing/credits at "
                    "https://platform.openai.com/account/billing then re-run the pilot."
                ) from exc
            wait = min(2**attempt, 20)
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

    raise RuntimeError(f"Failed to classify review after retries: {last_error}")


def analyze_dataframe(df: pd.DataFrame, client: OpenAI) -> tuple[pd.DataFrame, int, int]:
    rows: list[dict[str, Any]] = []
    success = 0
    failed = 0
    total = len(df)

    for idx, record in df.iterrows():
        logger.info("Classifying %s/%s ...", idx + 1, total)
        try:
            analysis = classify_review(client, record)
            row = record.to_dict()
            row.update(analysis.model_dump())
            rows.append(row)
            success += 1
        except Exception as exc:
            failed += 1
            logger.error("Skipping row %s due to error: %s", idx, exc)
            if "quota exceeded" in str(exc).lower() or "insufficient_quota" in str(exc).lower():
                logger.error("Stopping the pilot: OpenAI quota is exhausted.")
                break

        if CLASSIFY_SLEEP_SECONDS > 0 and idx < total - 1:
            time.sleep(CLASSIFY_SLEEP_SECONDS)

    if not rows:
        raise RuntimeError("No reviews were successfully analyzed.")

    result = pd.DataFrame(rows)
    return result, success, failed


def save_analyzed(df: pd.DataFrame, path: Path = OUTPUT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info("Wrote %s analyzed rows to %s", len(df), path)
    return path


def print_pilot_summary(df: pd.DataFrame, success: int, failed: int, output: Path) -> None:
    relevant = int(df["relevant"].sum()) if "relevant" in df.columns else 0
    irrelevant = int((~df["relevant"].astype(bool)).sum()) if "relevant" in df.columns else 0
    with_flavor = int(df["flavor"].notna().sum()) if "flavor" in df.columns else 0
    if "flavor" in df.columns:
        with_flavor = int(
            df["flavor"].fillna("").astype(str).str.strip().replace("None", "").ne("").sum()
        )

    print("=== Flavor Scout Step 3 pilot (50 reviews) ===")
    print(f"Successfully classified: {success}")
    print(f"Failed / skipped:       {failed}")
    print(f"Relevant:                {relevant}")
    print(f"Irrelevant:              {irrelevant}")
    print(f"With extracted flavor:   {with_flavor}")
    print("Sentiment distribution:")
    print(df["sentiment"].value_counts(dropna=False).to_string() if "sentiment" in df.columns else "  (n/a)")
    print("Intent distribution:")
    print(df["intent"].value_counts(dropna=False).to_string() if "intent" in df.columns else "  (n/a)")
    print(f"Final columns:           {', '.join(df.columns)}")
    print(f"Saved to:                {output}")
    print("\nFive representative examples:")
    cols = [
        c
        for c in [
            "product_title",
            "relevant",
            "flavor",
            "sentiment",
            "intent",
            "pain_point",
            "brand_fit",
            "confidence",
            "reasoning",
        ]
        if c in df.columns
    ]
    preview = df[cols].head(5).copy()
    if "product_title" in preview.columns:
        preview["product_title"] = preview["product_title"].astype(str).str.slice(0, 50)
    if "reasoning" in preview.columns:
        preview["reasoning"] = preview["reasoning"].astype(str).str.slice(0, 140)
    print(preview.to_string(index=False))


def run(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
    limit: int = CLASSIFY_PILOT_LIMIT,
) -> tuple[Path, pd.DataFrame, int, int]:
    client = load_openai_client()
    reviews = load_reviews(input_path, limit=limit)
    logger.info("Loaded %s reviews for pilot classification (limit=%s)", len(reviews), limit)
    analyzed, success, failed = analyze_dataframe(reviews, client)
    out = save_analyzed(analyzed, output_path)
    return out, analyzed, success, failed


def main() -> None:
    try:
        output, analyzed, success, failed = run()
        print_pilot_summary(analyzed, success, failed, output)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("Classification failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
