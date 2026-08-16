"""
Step 3: LLM classification of cleaned Amazon reviews.
Pilots:
  python -m src.classification.classify_reviews
  python -m src.classification.classify_reviews --validation-v2
Full run (checkpointed / resumable):
  python -m src.classification.classify_reviews --full
Batch-size benchmark (100 unclassified reviews, separate output):
  python -m src.classification.classify_reviews --batch-benchmark
OpenAI cost/performance benchmark (100 unclassified reviews, separate output):
  python -m src.classification.classify_reviews --openai-benchmark
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from openai import APIError, APITimeoutError, AsyncOpenAI, OpenAI, RateLimitError
from pydantic import ValidationError

from config import (
    CLASSIFY_ASYNC_CONCURRENCY,
    CLASSIFY_BATCH_BENCHMARK_LIMIT,
    CLASSIFY_BATCH_SIZE,
    CLASSIFY_CHECKPOINT_EVERY,
    CLASSIFY_COST_STOP_USD,
    CLASSIFY_FULL_SLEEP_SECONDS,
    CLASSIFY_MAX_PROMPT_TOKENS_PER_BATCH,
    CLASSIFY_MAX_RETRIES,
    CLASSIFY_OPENAI_BATCH_SIZE_TEST,
    CLASSIFY_OPENAI_MAX_PROMPT_TOKENS_PER_BATCH,
    CLASSIFY_PILOT_LIMIT,
    CLASSIFY_PILOT_SEED,
    CLASSIFY_SLEEP_SECONDS,
    CLASSIFY_TPM_BUDGET_FRACTION,
    CLASSIFY_TPM_LIMIT,
    GROQ_MODEL_FULL,
    OPENAI_INPUT_PRICE_PER_MILLION,
    OPENAI_MODEL_BENCHMARK,
    OPENAI_OUTPUT_PRICE_PER_MILLION,
    OPENAI_TPM_LIMIT,
)
from src.classification.llm_client import build_async_chat_client, build_chat_client
from src.classification.schemas import ReviewAnalysis
from src.cleaning.product_filters import is_gear_container_sku

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "processed_reviews.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "analyzed_reviews_pilot.csv"
OUTPUT_PATH_V2 = PROJECT_ROOT / "data" / "processed" / "analyzed_reviews_pilot_v2.csv"
OUTPUT_PATH_FULL = PROJECT_ROOT / "data" / "processed" / "analyzed_reviews.csv"
STATS_PATH_FULL = PROJECT_ROOT / "data" / "processed" / "analyzed_reviews_stats.json"
OUTPUT_PATH_BATCH_TEST = PROJECT_ROOT / "data" / "processed" / "analyzed_reviews_batch_test.csv"
STATS_PATH_BATCH_TEST = PROJECT_ROOT / "data" / "processed" / "analyzed_reviews_batch_test_stats.json"
OUTPUT_PATH_OPENAI_BENCH = PROJECT_ROOT / "data" / "processed" / "openai_100_benchmark.csv"
STATS_PATH_OPENAI_BENCH = PROJECT_ROOT / "data" / "processed" / "openai_100_benchmark_stats.json"
OUTPUT_PATH_OPENAI_BATCH8 = PROJECT_ROOT / "data" / "processed" / "openai_batch8_benchmark.csv"
STATS_PATH_OPENAI_BATCH8 = PROJECT_ROOT / "data" / "processed" / "openai_batch8_benchmark_stats.json"
OUTPUT_PATH_OPENAI_QUALITY = PROJECT_ROOT / "data" / "processed" / "openai_batch8_quality_validation.csv"
STATS_PATH_OPENAI_QUALITY = PROJECT_ROOT / "data" / "processed" / "openai_batch8_quality_validation_stats.json"
OUTPUT_PATH_OPENAI_FINAL_QUALITY = PROJECT_ROOT / "data" / "processed" / "openai_batch8_final_quality_validation.csv"
STATS_PATH_OPENAI_FINAL_QUALITY = PROJECT_ROOT / "data" / "processed" / "openai_batch8_final_quality_validation_stats.json"
OUTPUT_PATH_OPENAI_ASYNC = PROJECT_ROOT / "data" / "processed" / "openai_async_parallel_benchmark.csv"
STATS_PATH_OPENAI_ASYNC = PROJECT_ROOT / "data" / "processed" / "openai_async_parallel_benchmark_stats.json"

PROTECTED_OUTPUTS = {OUTPUT_PATH.resolve(), OUTPUT_PATH_V2.resolve()}
EXISTING_CLASSIFIED_OUTPUTS = {
    OUTPUT_PATH.resolve(),
    OUTPUT_PATH_V2.resolve(),
    OUTPUT_PATH_FULL.resolve(),
    OUTPUT_PATH_BATCH_TEST.resolve(),
    OUTPUT_PATH_OPENAI_BENCH.resolve(),
    OUTPUT_PATH_OPENAI_BATCH8.resolve(),
    OUTPUT_PATH_OPENAI_QUALITY.resolve(),
    OUTPUT_PATH_OPENAI_FINAL_QUALITY.resolve(),
}

EMPTY_ANALYSIS = {
    "relevant": pd.NA,
    "flavor": pd.NA,
    "sentiment": pd.NA,
    "intent": pd.NA,
    "pain_point": pd.NA,
    "brand_fit": pd.NA,
    "reasoning": pd.NA,
    "confidence": pd.NA,
}

NON_LAUNCHABLE_FLAVORS = {
    "garlic",
    "turkey dinner",
    "turkey",
    "scrambled eggs",
    "scrambled egg",
    "mashed potatoes",
    "mashed potato",
    "garlic mashed potatoes",
    "homestyle turkey dinner",
    "sweet pork",
    "homestyle chicken",
    "homestyle turkey",
    "chicken dinner",
    "pork dinner",
    "beef stew",
    "chili mac",
}

_MEAL_FLAVOR_MARKERS = (
    "dinner",
    "lunch",
    "breakfast",
    "homestyle",
    "entrée",
    "entree",
    "casserole",
    "mashed",
    "scrambled",
)

SYSTEM_PROMPT = """
You are Flavor Scout's classification analyst for HealthKart
(MuscleBlaze, HK Vitals, TrueBasics).
Flavor Scout's job is to find commercially meaningful NEW flavor opportunities
that HealthKart could launch — not to catalog every food word in Amazon reviews.
Classify ONE Amazon consumer review. Reason over the data; do not only summarize.
Internal order:
Raw review -> Relevant? -> Flavor? -> Sentiment? -> Consumer intent? -> Pain point? -> Brand fit?
Return ONLY valid JSON:
{
  "relevant": boolean,
  "flavor": string or null,
  "sentiment": "positive" | "neutral" | "negative",
  "intent": "request" | "preference" | "complaint" | "praise" | "comparison" | "general_mention" | "purchase_intent",
  "pain_point": string or null,
  "brand_fit": "strong" | "moderate" | "weak" | "none",
  "reasoning": string,
  "confidence": "high" | "medium" | "low"
}
RELEVANCE (assignment definition):
Judge the PRODUCT BEING REVIEWED (product_title / brand / categories / description),
not incidental words in the review text.
relevant=true ONLY when that SKU itself is a consumable in a Flavor Scout category:
food, beverage, protein, supplement, electrolyte drink/tablet, gummies,
wellness/functional nutrition, or another clearly ingestible product.
Otherwise relevant=false and brand_fit=none.
Do NOT mark relevant merely because:
- a bottle contains a drink
- a bag or cooler carries food/meals
- a backpack is used for hydration
- gear is used during exercise
- the reviewer mentions food, protein, or meals in passing
Normally IRRELEVANT (the reviewed SKU is the container/gear, not the consumable):
sport bottles, shaker bottles, tumblers, lunch bags, cooler bags, hydration
backpacks, hiking/tactical packs, fanny packs, foam rollers, clothing,
camping equipment, sports accessories, belts, bladders-as-containers.
EXCEPTION: if the reviewed SKU is itself a relevant consumable, keep relevant=true
even if the review also mentions gear.
Do not invent relevance. Do not stretch hydration gear into a flavor opportunity.
A freeze-dried meal SKU can be relevant as food even if flavor=null.
FLAVOR (launchable HealthKart flavor concept ONLY):
Set flavor only if the review explicitly mentions a flavor concept that could
reasonably become a product flavor (shake, whey, electrolyte, gummy, etc.).
Precision matters more than extracting many flavors.
Examples that SHOULD be extracted:
- "Chocolate protein tastes great" -> Chocolate
- "Watermelon electrolyte tastes refreshing" -> Watermelon
- "Masala chai whey would be amazing" -> Masala Chai
- "Kesar pista protein should exist" -> Kesar Pista
- "Blueberry gummies taste great" -> Blueberry
- "Citrus Berry tablets" -> Citrus Berry
Examples that must be flavor=null (keep the review; do not delete it):
- "Garlic mashed potatoes taste good" -> flavor=null  (NEVER Garlic)
- "Mountain House turkey dinner was great" -> flavor=null
- "Sweet Pork" / "Homestyle Chicken" freeze-dried meals -> flavor=null
- scrambled eggs, mashed potatoes, complete dinners/entrées, dish names,
  product meal names, random ingredients that are not plausible HealthKart flavors
Never invent a flavor. A meal/entrée/product name is not a flavor.
Positive or negative mentions of a real flavor still count.
SENTIMENT: positive | neutral | negative
INTENT (primary):
request | preference | complaint | praise | comparison | general_mention
Use purchase_intent when clearly supported ("I'd buy this", "I wish this existed",
"Would definitely try", "Please launch this").
PAIN POINT: genuine stated problem, else null. Do not invent one.
BRAND FIT:
brand_fit is a FIT SCORE, not a brand name.
Allowed values ONLY: strong | moderate | weak | none
NEVER set brand_fit to MuscleBlaze, HK Vitals, TrueBasics, HealthKart, or any SKU.
If a brand is relevant, name it in reasoning and still set brand_fit to a score.
MuscleBlaze context = performance / gym / protein / bold flavors
HK Vitals context = wellness / lifestyle / everyday health
TrueBasics context = premium wellness / functional nutrition
Use none if the review is not useful for those brands. Do not invent unsupported brand/SKU names.
REASONING: concise evidence-based justification for the labels.
CONFIDENCE: high | medium | low
""".strip()

BATCH_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    "Classify ONE Amazon consumer review. Reason over the data; do not only summarize.",
    "Classify EACH Amazon consumer review independently. Reason over each review; "
    "do not summarize the batch and do not copy one classification onto another review.",
).replace(
    'Return ONLY valid JSON:\n{\n  "relevant": boolean,\n  "flavor": string or null,\n  '
    '"sentiment": "positive" | "neutral" | "negative",\n  "intent": "request" | "preference" | '
    '"complaint" | "praise" | "comparison" | "general_mention" | "purchase_intent",\n  '
    '"pain_point": string or null,\n  "brand_fit": "strong" | "moderate" | "weak" | "none",\n  '
    '"reasoning": string,\n  "confidence": "high" | "medium" | "low"\n}',
    """Return ONLY valid JSON of this shape:
{
  "classifications": [
    {
      "review_key": string,
      "relevant": boolean,
      "flavor": string or null,
      "sentiment": "positive" | "neutral" | "negative",
      "intent": "request" | "preference" | "complaint" | "praise" | "comparison" | "general_mention" | "purchase_intent",
      "pain_point": string or null,
      "brand_fit": "strong" | "moderate" | "weak" | "none",
      "reasoning": string,
      "confidence": "high" | "medium" | "low"
    }
  ]
}
Every input review_key must appear exactly once. Classify independently per review.""",
)

def review_key(record: pd.Series | dict[str, Any]) -> str:
    payload = "|".join(
        [
            str(record.get("product_id", "")),
            str(record.get("parent_asin", "")),
            str(record.get("timestamp", "")),
            str(record.get("review_text", "")),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()[:20]

def load_reviews(
    path: Path = INPUT_PATH,
    limit: int | None = CLASSIFY_PILOT_LIMIT,
    random_sample: bool = False,
    seed: int = CLASSIFY_PILOT_SEED,
) -> pd.DataFrame:
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
    if random_sample:
        n = min(limit or len(df), len(df))
        return df.sample(n=n, random_state=seed).reset_index(drop=True)
    if limit is None:
        return df.reset_index(drop=True)
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
    return (
        "Classify this Amazon review for Flavor Scout.\n\n"
        f"product_title: {record.get('product_title') or ''}\n"
        f"brand: {record.get('brand') or ''}\n"
        f"main_category: {record.get('main_category') or ''}\n"
        f"product_categories: {record.get('product_categories') or ''}\n"
        f"product_description: {str(record.get('product_description') or '')[:800]}\n"
        f"review_title: {record.get('review_title') or ''}\n"
        f"review_text: {record.get('review_text') or ''}"
    )

def _estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)

def _batch_item_payload(record: pd.Series, key: str) -> str:
    return (
        f"review_key: {key}\n"
        f"product_title: {record.get('product_title') or ''}\n"
        f"brand: {record.get('brand') or ''}\n"
        f"main_category: {record.get('main_category') or ''}\n"
        f"product_categories: {record.get('product_categories') or ''}\n"
        f"product_description: {str(record.get('product_description') or '')[:800]}\n"
        f"review_title: {record.get('review_title') or ''}\n"
        f"review_text: {record.get('review_text') or ''}"
    )

def _batch_user_payload(items: list[tuple[str, pd.Series]]) -> str:
    parts = [
        "Classify each of the following Amazon reviews independently for Flavor Scout.",
        "Return one classification object per review_key. Do not summarize the batch.",
        "",
    ]
    for i, (key, record) in enumerate(items, start=1):
        parts.append(f"--- REVIEW {i} ---")
        parts.append(_batch_item_payload(record, key))
        parts.append("")
    return "\n".join(parts)

def _empty_stats() -> dict[str, int]:
    return {
        "success": 0,
        "failed": 0,
        "http_429": 0,
        "retries": 0,
        "quota_stop": 0,
        "excluded_non_launchable": 0,
        "resumed": 0,
        "api_requests": 0,
        "batch_requests": 0,
        "single_fallback_requests": 0,
        "pydantic_failures": 0,
        "pydantic_passed": 0,
        "tokens_used": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "request_latency_sum": 0.0,
        "request_latency_count": 0,
        "estimated_cost_usd": 0.0,
        "cost_stop": 0,
        "sku_relevance_overrides": 0,
    }

def estimate_openai_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        (prompt_tokens / 1_000_000.0) * OPENAI_INPUT_PRICE_PER_MILLION
        + (completion_tokens / 1_000_000.0) * OPENAI_OUTPUT_PRICE_PER_MILLION
    )

def _refresh_cost(stats: dict[str, Any]) -> float:
    cost = estimate_openai_cost(
        int(stats.get("prompt_tokens") or 0),
        int(stats.get("completion_tokens") or 0),
    )
    stats["estimated_cost_usd"] = round(cost, 6)
    return cost

def _pace_after_request(
    tokens_used: int,
    min_sleep: float,
    tpm_limit: int = CLASSIFY_TPM_LIMIT,
) -> None:
    budget = tpm_limit * CLASSIFY_TPM_BUDGET_FRACTION
    tpm_wait = (tokens_used / budget) * 60.0 if budget > 0 and tokens_used > 0 else 0.0
    wait = max(min_sleep, tpm_wait) + random.uniform(0.05, 0.35)
    logger.info("Pacing %.1fs after %s tokens (TPM budget %.0f)", wait, tokens_used, budget)
    time.sleep(wait)

def _chat_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_content: str,
    stats: dict[str, int],
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, CLASSIFY_MAX_RETRIES + 1):
        try:
            stats["api_requests"] = stats.get("api_requests", 0) + 1
            started = time.perf_counter()
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            latency = time.perf_counter() - started
            stats["request_latency_sum"] = float(stats.get("request_latency_sum") or 0.0) + latency
            stats["request_latency_count"] = int(stats.get("request_latency_count") or 0) + 1
            usage = getattr(response, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            total_tokens = int(getattr(usage, "total_tokens", 0) or 0) or (
                prompt_tokens + completion_tokens
            )
            stats["prompt_tokens"] = int(stats.get("prompt_tokens") or 0) + prompt_tokens
            stats["completion_tokens"] = int(stats.get("completion_tokens") or 0) + completion_tokens
            stats["tokens_used"] = stats.get("tokens_used", 0) + total_tokens
            stats["_last_tokens"] = total_tokens
            content = response.choices[0].message.content or ""
            return _extract_json_payload(content)
        except (RateLimitError, APITimeoutError, APIError) as exc:
            last_error = exc
            message = str(exc).lower()
            if "insufficient_quota" in message:
                raise RuntimeError(
                    "LLM provider quota exceeded. Check provider credits, then resume --full."
                ) from exc
            if _is_rate_limit(exc):
                stats["http_429"] = stats.get("http_429", 0) + 1
            stats["retries"] = stats.get("retries", 0) + 1
            wait = _backoff_seconds(attempt, _retry_after_seconds(exc))
            logger.warning(
                "LLM API error on attempt %s/%s: %s. Waiting %.1fs then retrying.",
                attempt,
                CLASSIFY_MAX_RETRIES,
                exc,
                wait,
            )
            time.sleep(wait)
        except (json.JSONDecodeError, TypeError, KeyError, IndexError) as exc:
            last_error = exc
            stats["retries"] = stats.get("retries", 0) + 1
            logger.warning(
                "Invalid AI response on attempt %s/%s: %s",
                attempt,
                CLASSIFY_MAX_RETRIES,
                exc,
            )
            time.sleep(1 + random.uniform(0.0, 0.5))
    raise RuntimeError(f"Failed LLM JSON call after retries: {last_error}")

def _iter_batches(
    pending: list[tuple[str, pd.Series]],
    batch_size: int,
    max_prompt_tokens: int,
) -> list[list[tuple[str, pd.Series]]]:
    batches: list[list[tuple[str, pd.Series]]] = []
    current: list[tuple[str, pd.Series]] = []
    prompt_tokens = _estimate_tokens(BATCH_SYSTEM_PROMPT)
    for key, record in pending:
        item_tokens = _estimate_tokens(_batch_item_payload(record, key)) + 40
        would_exceed = current and (
            len(current) >= batch_size or prompt_tokens + item_tokens > max_prompt_tokens
        )
        if would_exceed:
            batches.append(current)
            current = []
            prompt_tokens = _estimate_tokens(BATCH_SYSTEM_PROMPT)
        current.append((key, record))
        prompt_tokens += item_tokens
    if current:
        batches.append(current)
    return batches

def _extract_classification_dicts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    for key in ("classifications", "results", "reviews", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    if "review_key" in payload and "relevant" in payload:
        return [payload]
    return []

def classify_batch(
    client: OpenAI,
    items: list[tuple[str, pd.Series]],
    model: str,
    stats: dict[str, int],
) -> dict[str, ReviewAnalysis]:
    """Return independently validated classifications keyed by review_key."""
    parsed: dict[str, ReviewAnalysis] = {}
    expected = {key for key, _ in items}
    stats["batch_requests"] = stats.get("batch_requests", 0) + 1
    try:
        payload = _chat_json(
            client,
            model,
            BATCH_SYSTEM_PROMPT,
            _batch_user_payload(items),
            stats,
        )
    except RuntimeError as exc:
        logger.warning("Batch request failed; falling back to single-review calls: %s", exc)
        return parsed

    seen: set[str] = set()
    for raw in _extract_classification_dicts(payload):
        key = str(raw.get("review_key") or "").strip()
        if not key or key not in expected:
            stats["pydantic_failures"] = stats.get("pydantic_failures", 0) + 1
            logger.warning("Dropping batch item with unknown or missing review_key=%s", key or "<empty>")
            continue
        if key in seen:
            logger.warning("Duplicate classification for review_key=%s; keeping first", key)
            continue
        seen.add(key)
        fields = {k: v for k, v in raw.items() if k != "review_key"}
        try:
            parsed[key] = ReviewAnalysis.model_validate(fields)
            stats["pydantic_passed"] = stats.get("pydantic_passed", 0) + 1
        except ValidationError as exc:
            stats["pydantic_failures"] = stats.get("pydantic_failures", 0) + 1
            logger.warning("Pydantic failed for review_key=%s: %s", key, exc)
    missing = expected - parsed.keys()
    if missing:
        logger.warning(
            "Batch returned %s/%s valid classifications; %s will retry individually",
            len(parsed),
            len(expected),
            len(missing),
        )
    return parsed

def _is_rate_limit(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    text = str(exc).lower()
    return "429" in text or "too many requests" in text or "rate limit" in text

def _retry_after_seconds(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    raw = None
    if hasattr(headers, "get"):
        raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is not None:
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            pass
    match = re.search(r"try again in\s+([\d.]+)\s*s", str(exc), flags=re.IGNORECASE)
    if match:
        return max(0.0, float(match.group(1)))
    return None

def _backoff_seconds(attempt: int, retry_after: float | None) -> float:
    if retry_after is not None:
        return retry_after + random.uniform(0.05, 0.4)
    base = min(2**attempt, 30)
    return base + random.uniform(0.1, 1.0)

def classify_review(
    client: OpenAI,
    record: pd.Series,
    model: str,
    stats: dict[str, int],
) -> ReviewAnalysis:
    stats["single_fallback_requests"] = stats.get("single_fallback_requests", 0) + 1
    last_error: Exception | None = None
    user_content = _user_payload(record)
    for attempt in range(1, CLASSIFY_MAX_RETRIES + 1):
        try:
            payload = _chat_json(client, model, SYSTEM_PROMPT, user_content, stats)
            analysis = ReviewAnalysis.model_validate(payload)
            stats["pydantic_passed"] = stats.get("pydantic_passed", 0) + 1
            return analysis
        except RuntimeError as exc:
            last_error = exc
            raise
        except ValidationError as exc:
            last_error = exc
            stats["pydantic_failures"] = stats.get("pydantic_failures", 0) + 1
            stats["retries"] = stats.get("retries", 0) + 1
            logger.warning(
                "Invalid AI response on attempt %s/%s: %s",
                attempt,
                CLASSIFY_MAX_RETRIES,
                exc,
            )
            time.sleep(1 + random.uniform(0.0, 0.5))
    raise RuntimeError(f"Failed to classify review after retries: {last_error}")

def _normalize_flavor(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    return text

def _is_non_launchable_flavor(flavor: str) -> bool:
    lowered = flavor.strip().lower()
    if lowered in NON_LAUNCHABLE_FLAVORS:
        return True
    return any(marker in lowered for marker in _MEAL_FLAVOR_MARKERS)

def apply_launchable_flavor_qc(row: dict[str, Any], stats: dict[str, int]) -> dict[str, Any]:
    flavor = _normalize_flavor(row.get("flavor"))
    if flavor and _is_non_launchable_flavor(flavor):
        stats["excluded_non_launchable"] += 1
        row["flavor"] = None
    else:
        row["flavor"] = flavor
    return row

def apply_sku_relevance_guard(row: dict[str, Any], stats: dict[str, int]) -> dict[str, Any]:
    if not is_gear_container_sku(row):
        return row
    if row.get("relevant") is True or _normalize_flavor(row.get("flavor")):
        stats["sku_relevance_overrides"] = int(stats.get("sku_relevance_overrides") or 0) + 1
    row["relevant"] = False
    row["flavor"] = None
    row["brand_fit"] = "none"
    return row

def load_checkpoint(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    existing = pd.read_csv(path)
    logger.info("Resuming from checkpoint %s (%s rows)", path, len(existing))
    return existing

def load_prior_classifications() -> pd.DataFrame:
    """Merge successful rows from all classified outputs without re-calling the LLM."""
    frames: list[pd.DataFrame] = []
    for path in (OUTPUT_PATH_FULL, OUTPUT_PATH_BATCH_TEST, OUTPUT_PATH_OPENAI_BENCH):
        if not path.exists():
            continue
        part = pd.read_csv(path)
        logger.info("Loaded prior classifications from %s (%s rows)", path.name, len(part))
        frames.append(part)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    if "review_key" not in merged.columns:
        merged["review_key"] = merged.apply(review_key, axis=1)
    if "classification_status" in merged.columns:
        merged["_ok"] = (merged["classification_status"].astype(str) == "success").astype(int)
        merged = merged.sort_values("_ok", ascending=False)
        merged = merged.drop_duplicates(subset=["review_key"], keep="first")
        merged = merged.drop(columns=["_ok"])
    else:
        merged = merged.drop_duplicates(subset=["review_key"], keep="first")
    return merged.reset_index(drop=True)

def save_checkpoint(rows: list[dict[str, Any]], path: Path) -> None:
    if path.resolve() in PROTECTED_OUTPUTS:
        raise RuntimeError(f"Refusing to write checkpoint over protected file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")

def save_stats(stats: dict[str, Any], path: Path = STATS_PATH_FULL) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for key, value in stats.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            serializable[key] = value
        else:
            serializable[key] = str(value)
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

def analyze_dataframe(
    df: pd.DataFrame,
    client: OpenAI,
    model: str,
    output_path: Path | None = None,
    sleep_seconds: float = CLASSIFY_SLEEP_SECONDS,
    checkpoint_every: int = CLASSIFY_CHECKPOINT_EVERY,
    existing: pd.DataFrame | None = None,
    stats_path: Path | None = None,
    batch_size: int = 1,
    skip_keys: set[str] | None = None,
    pace_for_tpm: bool = False,
    tpm_limit: int = CLASSIFY_TPM_LIMIT,
    cost_stop_usd: float | None = None,
    max_prompt_tokens: int = CLASSIFY_MAX_PROMPT_TOKENS_PER_BATCH,
) -> tuple[pd.DataFrame, dict[str, int]]:
    done_keys: set[str] = set(skip_keys or set())
    rows: list[dict[str, Any]] = []
    if existing is not None and not existing.empty:
        for _, prev in existing.iterrows():
            key = str(prev.get("review_key") or review_key(prev))
            status = str(prev.get("classification_status") or "")
            if status == "success" or key in done_keys:
                done_keys.add(key)
                rows.append(prev.to_dict())

    stats = _empty_stats()
    stats["success"] = int(sum(1 for r in rows if r.get("classification_status") == "success"))
    stats["failed"] = int(sum(1 for r in rows if r.get("classification_status") == "failed"))
    stats["resumed"] = len(rows)
    stats["batch_size_configured"] = int(batch_size)
    total = len(df)

    pending: list[tuple[str, pd.Series, int]] = []
    for idx, record in df.iterrows():
        key = review_key(record)
        if key in done_keys:
            continue
        pending.append((key, record, int(idx) + 1))

    def _persist() -> None:
        if output_path is not None:
            save_checkpoint(rows, output_path)
            if stats_path is not None:
                save_stats(stats, stats_path)

    def _record_success(key: str, record: pd.Series, analysis: ReviewAnalysis) -> None:
        base = record.to_dict()
        base["review_key"] = key
        row = {**base, **analysis.model_dump()}
        row["classification_status"] = "success"
        row["error_message"] = ""
        row = apply_launchable_flavor_qc(row, stats)
        row = apply_sku_relevance_guard(row, stats)
        rows.append(row)
        stats["success"] += 1
        done_keys.add(key)

    def _record_failure(key: str, record: pd.Series, exc: Exception) -> None:
        base = record.to_dict()
        base["review_key"] = key
        stats["failed"] += 1
        logger.error("Recording failed row %s: %s", key, exc)
        row = {**base, **EMPTY_ANALYSIS}
        row["classification_status"] = "failed"
        row["error_message"] = str(exc)
        rows.append(row)
        done_keys.add(key)

    if batch_size <= 1:
        newly_done = 0
        for key, record, position in pending:
            logger.info("Classifying %s/%s ...", position, total)
            try:
                analysis = classify_review(client, record, model=model, stats=stats)
                _record_success(key, record, analysis)
            except Exception as exc:
                _record_failure(key, record, exc)
                if "quota exceeded" in str(exc).lower() or "insufficient_quota" in str(exc).lower():
                    stats["quota_stop"] = 1
                    logger.error("Quota exhausted; checkpointing and stopping.")
                    break
            newly_done += 1
            if output_path is not None and newly_done % checkpoint_every == 0:
                _persist()
                logger.info("Checkpoint saved: %s classified rows", len(rows))
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
        _persist()
        result = pd.DataFrame(rows)
        stats["tested"] = len(result)
        return result, stats

    work = [(key, record) for key, record, _ in pending]
    batches = _iter_batches(work, batch_size, max_prompt_tokens)
    stats["batch_count"] = len(batches)
    logger.info(
        "Batching %s remaining reviews into %s requests (max batch_size=%s)",
        len(work),
        len(batches),
        batch_size,
    )

    quota_stop = False
    for batch_idx, batch in enumerate(batches, start=1):
        if quota_stop:
            break
        logger.info(
            "Classifying batch %s/%s (%s reviews) ...",
            batch_idx,
            len(batches),
            len(batch),
        )
        parsed = classify_batch(client, batch, model=model, stats=stats)
        if pace_for_tpm:
            _pace_after_request(
                int(stats.get("_last_tokens") or 1800),
                sleep_seconds,
                tpm_limit=tpm_limit,
            )

        for key, record in batch:
            if key in parsed:
                _record_success(key, record, parsed[key])
                continue
            logger.info("Retrying review_key=%s as a single request", key)
            try:
                analysis = classify_review(client, record, model=model, stats=stats)
                _record_success(key, record, analysis)
                if pace_for_tpm:
                    _pace_after_request(
                        int(stats.get("_last_tokens") or 1200),
                        sleep_seconds,
                        tpm_limit=tpm_limit,
                    )
            except Exception as exc:
                _record_failure(key, record, exc)
                if "quota exceeded" in str(exc).lower() or "insufficient_quota" in str(exc).lower():
                    stats["quota_stop"] = 1
                    quota_stop = True
                    logger.error("Quota exhausted; checkpointing and stopping.")
                    break

        _persist()
        cost = _refresh_cost(stats)
        logger.info(
            "Checkpoint saved after batch %s: %s classified rows | est. cost $%.4f",
            batch_idx,
            len(rows),
            cost,
        )
        if cost_stop_usd is not None and cost >= cost_stop_usd:
            stats["cost_stop"] = 1
            logger.error(
                "Cost safety stop at $%.4f (limit $%.2f). Checkpoint preserved; resume --full later.",
                cost,
                cost_stop_usd,
            )
            break

    result = pd.DataFrame(rows)
    stats["tested"] = len(result)
    classified_now = stats["success"] + stats["failed"] - stats["resumed"]
    stats["reviews_processed_this_run"] = max(0, classified_now)
    return result, stats

def save_analyzed(df: pd.DataFrame, path: Path) -> Path:
    if path.resolve() in PROTECTED_OUTPUTS and path.exists() and path.name != Path(sys.argv[0]).name:

        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info("Wrote %s analyzed rows to %s", len(df), path)
    return path

def _nonzero_text_count(series: pd.Series) -> int:
    return int(series.fillna("").astype(str).str.strip().replace({"None": "", "nan": ""}).ne("").sum())

def flavor_qc_report(df: pd.DataFrame) -> dict[str, Any]:
    classified = df[df["classification_status"] == "success"] if "classification_status" in df.columns else df
    flavors = (
        classified["flavor"].fillna("").astype(str).str.strip()
        if "flavor" in classified.columns
        else pd.Series(dtype=str)
    )
    flavors = flavors[flavors.ne("") & flavors.str.lower().ne("none") & flavors.str.lower().ne("nan")]
    banned_hits = flavors[flavors.map(lambda x: _is_non_launchable_flavor(str(x)))]
    return {
        "launchable_flavor_mentions": int(len(flavors)),
        "unique_flavor_concepts": int(flavors.str.lower().nunique()) if len(flavors) else 0,
        "banned_flavor_hits": int(len(banned_hits)),
        "banned_flavor_values": sorted(set(banned_hits.str.lower().tolist())),
        "top_flavors": flavors.str.lower().value_counts().head(15).to_dict() if len(flavors) else {},
    }

def print_full_summary(
    df: pd.DataFrame,
    stats: dict[str, Any],
    output: Path,
    provider: str,
    model: str,
    duration_seconds: float,
) -> None:
    classified = df[df["classification_status"] == "success"] if "classification_status" in df.columns else df
    relevant = int(classified["relevant"].fillna(False).astype(bool).sum()) if "relevant" in classified.columns else 0
    irrelevant = int((~classified["relevant"].fillna(False).astype(bool)).sum()) if "relevant" in classified.columns else 0
    qc = flavor_qc_report(df)
    pain_count = _nonzero_text_count(classified["pain_point"]) if "pain_point" in classified.columns else 0
    purchase = int((classified["intent"].astype(str).str.lower() == "purchase_intent").sum()) if "intent" in classified.columns else 0

    print("=== Flavor Scout Step 3 FULL classification ===")
    print(f"LLM provider:                 {provider}")
    print(f"Model:                        {model}")
    print(f"Total reviews:                {stats.get('tested', len(df))}")
    print(f"Successful classifications:   {stats.get('success', 0)}")
    print(f"Failed classifications:       {stats.get('failed', 0)}")
    print(f"Relevant reviews:             {relevant}")
    print(f"Irrelevant reviews:           {irrelevant}")
    print(f"Launchable flavor mentions:   {qc['launchable_flavor_mentions']}")
    print(f"Unique flavor concepts:       {qc['unique_flavor_concepts']}")
    print(f"Excluded non-launchable terms:{stats.get('excluded_non_launchable', 0)}")
    print(f"Banned flavors still present: {qc['banned_flavor_hits']} {qc['banned_flavor_values']}")
    print(f"Purchase-intent count:        {purchase}")
    print(f"Pain-point count:             {pain_count}")
    print(f"HTTP 429 responses:           {stats.get('http_429', 0)}")
    print(f"Retries:                      {stats.get('retries', 0)}")
    print(f"Pydantic passed:              {stats.get('pydantic_passed', 0)}")
    print(f"Pydantic failures:            {stats.get('pydantic_failures', 0)}")
    print(f"API requests:                 {stats.get('api_requests', 0)}")
    print(f"Input tokens:                 {stats.get('prompt_tokens', 0)}")
    print(f"Output tokens:                {stats.get('completion_tokens', 0)}")
    print(f"Total tokens:                 {stats.get('tokens_used', 0)}")
    print(f"Estimated API cost (USD):     ${float(stats.get('estimated_cost_usd') or 0):.4f}")
    print(f"Cost safety stop:             {bool(stats.get('cost_stop'))}")
    print(f"Newly classified this run:    {stats.get('reviews_processed_this_run', 0)}")
    print(f"Resumed prior rows:           {stats.get('resumed', 0)}")
    print(f"Processing duration (sec):    {duration_seconds:.1f}")
    print("Sentiment distribution:")
    print(classified["sentiment"].value_counts(dropna=False).to_string() if "sentiment" in classified.columns else "  (n/a)")
    print("Intent distribution:")
    print(classified["intent"].value_counts(dropna=False).to_string() if "intent" in classified.columns else "  (n/a)")
    print("Brand-fit distribution:")
    print(classified["brand_fit"].value_counts(dropna=False).to_string() if "brand_fit" in classified.columns else "  (n/a)")
    print(f"Top flavors: {qc['top_flavors']}")
    print(f"Saved to:                     {output}")

def print_pilot_summary(
    df: pd.DataFrame,
    stats: dict[str, int],
    output: Path,
    provider: str,
    model: str,
) -> None:
    classified = df[df.get("classification_status", "success") == "success"] if "classification_status" in df.columns else df
    relevant = int(classified["relevant"].fillna(False).astype(bool).sum()) if "relevant" in classified.columns else 0
    irrelevant = int((~classified["relevant"].fillna(False).astype(bool)).sum()) if "relevant" in classified.columns else 0
    with_flavor = _nonzero_text_count(classified["flavor"]) if "flavor" in classified.columns else 0
    print("=== Flavor Scout Step 3 pilot ===")
    print(f"LLM provider:            {provider}")
    print(f"Model:                   {model}")
    print(f"Successfully classified: {stats.get('success', 0)}")
    print(f"Failed / skipped:        {stats.get('failed', 0)}")
    print(f"Relevant:                {relevant}")
    print(f"Irrelevant:              {irrelevant}")
    print(f"With extracted flavor:   {with_flavor}")
    print(f"Saved to:                {output}")

async def _achat_json(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    user_content: str,
    stats: dict[str, Any],
    lock: asyncio.Lock,
    semaphore: asyncio.Semaphore,
    stop_event: asyncio.Event | None = None,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, CLASSIFY_MAX_RETRIES + 1):
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("Cost safety stop reached; skipping remaining API calls.")
        try:
            async with semaphore:
                if stop_event is not None and stop_event.is_set():
                    raise RuntimeError("Cost safety stop reached; skipping remaining API calls.")
                started = time.perf_counter()
                response = await client.chat.completions.create(
                    model=model,
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                )
                latency = time.perf_counter() - started
            usage = getattr(response, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            total_tokens = int(getattr(usage, "total_tokens", 0) or 0) or (
                prompt_tokens + completion_tokens
            )
            async with lock:
                stats["api_requests"] = int(stats.get("api_requests") or 0) + 1
                stats["request_latency_sum"] = float(stats.get("request_latency_sum") or 0.0) + latency
                stats["request_latency_count"] = int(stats.get("request_latency_count") or 0) + 1
                stats["prompt_tokens"] = int(stats.get("prompt_tokens") or 0) + prompt_tokens
                stats["completion_tokens"] = int(stats.get("completion_tokens") or 0) + completion_tokens
                stats["tokens_used"] = int(stats.get("tokens_used") or 0) + total_tokens
                stats["_last_tokens"] = total_tokens
                cost = _refresh_cost(stats)
                if stop_event is not None and cost >= CLASSIFY_COST_STOP_USD:
                    stats["cost_stop"] = 1
                    stop_event.set()
            content = response.choices[0].message.content or ""
            return _extract_json_payload(content)
        except (RateLimitError, APITimeoutError, APIError) as exc:
            last_error = exc
            message = str(exc).lower()
            if "insufficient_quota" in message:
                raise RuntimeError(
                    "LLM provider quota exceeded. Check provider credits, then resume --full."
                ) from exc
            async with lock:
                if _is_rate_limit(exc):
                    stats["http_429"] = int(stats.get("http_429") or 0) + 1
                stats["retries"] = int(stats.get("retries") or 0) + 1
            wait = _backoff_seconds(attempt, _retry_after_seconds(exc))
            logger.warning(
                "Async LLM API error on attempt %s/%s: %s. Waiting %.1fs.",
                attempt,
                CLASSIFY_MAX_RETRIES,
                exc,
                wait,
            )
            await asyncio.sleep(wait)
        except (json.JSONDecodeError, TypeError, KeyError, IndexError) as exc:
            last_error = exc
            async with lock:
                stats["retries"] = int(stats.get("retries") or 0) + 1
            logger.warning(
                "Invalid async AI response on attempt %s/%s: %s",
                attempt,
                CLASSIFY_MAX_RETRIES,
                exc,
            )
            await asyncio.sleep(1 + random.uniform(0.0, 0.5))
    raise RuntimeError(f"Failed LLM JSON call after retries: {last_error}")

async def classify_batch_async(
    client: AsyncOpenAI,
    items: list[tuple[str, pd.Series]],
    model: str,
    stats: dict[str, Any],
    lock: asyncio.Lock,
    semaphore: asyncio.Semaphore,
    stop_event: asyncio.Event | None = None,
) -> dict[str, ReviewAnalysis]:
    parsed: dict[str, ReviewAnalysis] = {}
    expected = {key for key, _ in items}
    async with lock:
        stats["batch_requests"] = int(stats.get("batch_requests") or 0) + 1
    try:
        payload = await _achat_json(
            client,
            model,
            BATCH_SYSTEM_PROMPT,
            _batch_user_payload(items),
            stats,
            lock,
            semaphore,
            stop_event,
        )
    except RuntimeError as exc:
        logger.warning("Async batch request failed; falling back to single-review calls: %s", exc)
        return parsed
    seen: set[str] = set()
    for raw in _extract_classification_dicts(payload):
        key = str(raw.get("review_key") or "").strip()
        if not key or key not in expected or key in seen:
            async with lock:
                stats["pydantic_failures"] = int(stats.get("pydantic_failures") or 0) + 1
            continue
        seen.add(key)
        fields = {k: v for k, v in raw.items() if k != "review_key"}
        try:
            parsed[key] = ReviewAnalysis.model_validate(fields)
            async with lock:
                stats["pydantic_passed"] = int(stats.get("pydantic_passed") or 0) + 1
        except ValidationError as exc:
            async with lock:
                stats["pydantic_failures"] = int(stats.get("pydantic_failures") or 0) + 1
            logger.warning("Pydantic failed for review_key=%s: %s", key, exc)
    return parsed

async def classify_review_async(
    client: AsyncOpenAI,
    record: pd.Series,
    model: str,
    stats: dict[str, Any],
    lock: asyncio.Lock,
    semaphore: asyncio.Semaphore,
    stop_event: asyncio.Event | None = None,
) -> ReviewAnalysis:
    async with lock:
        stats["single_fallback_requests"] = int(stats.get("single_fallback_requests") or 0) + 1
    last_error: Exception | None = None
    user_content = _user_payload(record)
    for attempt in range(1, CLASSIFY_MAX_RETRIES + 1):
        try:
            payload = await _achat_json(
                client, model, SYSTEM_PROMPT, user_content, stats, lock, semaphore, stop_event
            )
            analysis = ReviewAnalysis.model_validate(payload)
            async with lock:
                stats["pydantic_passed"] = int(stats.get("pydantic_passed") or 0) + 1
            return analysis
        except RuntimeError:
            raise
        except ValidationError as exc:
            last_error = exc
            async with lock:
                stats["pydantic_failures"] = int(stats.get("pydantic_failures") or 0) + 1
                stats["retries"] = int(stats.get("retries") or 0) + 1
            await asyncio.sleep(1 + random.uniform(0.0, 0.5))
    raise RuntimeError(f"Failed to classify review after retries: {last_error}")

def _row_from_analysis(
    key: str,
    record: pd.Series,
    analysis: ReviewAnalysis,
    stats: dict[str, Any],
) -> dict[str, Any]:
    base = record.to_dict()
    base["review_key"] = key
    row = {**base, **analysis.model_dump()}
    row["classification_status"] = "success"
    row["error_message"] = ""
    row = apply_launchable_flavor_qc(row, stats)
    return apply_sku_relevance_guard(row, stats)

async def run_async_openai_benchmark(
    reviews: pd.DataFrame,
    output_path: Path,
    stats_path: Path,
    model: str,
    batch_size: int,
    max_prompt_tokens: int,
    concurrency: int,
    cost_stop_usd: float,
    existing: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    client, provider, model_id = build_async_chat_client(provider="openai", model=model)
    done_keys: set[str] = set()
    rows: list[dict[str, Any]] = []
    if existing is not None and not existing.empty:
        for _, prev in existing.iterrows():
            key = str(prev.get("review_key") or review_key(prev))
            if str(prev.get("classification_status") or "") == "success":
                done_keys.add(key)
                rows.append(prev.to_dict())
    pending = [
        (review_key(record), record)
        for _, record in reviews.iterrows()
        if review_key(record) not in done_keys
    ]
    batches = _iter_batches(pending, batch_size, max_prompt_tokens)
    stats = _empty_stats()
    stats["batch_size_configured"] = batch_size
    stats["async_concurrency"] = concurrency
    stats["batch_count"] = len(batches)
    stats["provider"] = provider
    stats["resumed"] = len(rows)
    stats["success"] = int(sum(1 for r in rows if r.get("classification_status") == "success"))
    stats["failed"] = 0
    lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(concurrency)
    stop_event = asyncio.Event()
    prior_count = len(rows)

    async def process_batch(batch_idx: int, batch: list[tuple[str, pd.Series]]) -> None:
        if stop_event.is_set():
            return
        logger.info("Async classifying batch %s/%s (%s reviews)", batch_idx, len(batches), len(batch))
        parsed = await classify_batch_async(
            client, batch, model_id, stats, lock, semaphore, stop_event
        )
        batch_rows: list[dict[str, Any]] = []
        for key, record in batch:
            if stop_event.is_set() and key not in parsed:
                break
            if key in parsed:
                batch_rows.append(_row_from_analysis(key, record, parsed[key], stats))
                continue
            logger.info("Async retrying review_key=%s as a single request", key)
            try:
                analysis = await classify_review_async(
                    client, record, model_id, stats, lock, semaphore, stop_event
                )
                batch_rows.append(_row_from_analysis(key, record, analysis, stats))
            except Exception as exc:
                if "Cost safety stop" in str(exc):
                    break
                base = record.to_dict()
                base["review_key"] = key
                row = {**base, **EMPTY_ANALYSIS}
                row["classification_status"] = "failed"
                row["error_message"] = str(exc)
                batch_rows.append(row)
        async with lock:
            rows.extend(batch_rows)
            stats["success"] = int(sum(1 for r in rows if r.get("classification_status") == "success"))
            stats["failed"] = int(sum(1 for r in rows if r.get("classification_status") == "failed"))
            cost = _refresh_cost(stats)
            save_checkpoint(rows, output_path)
            save_stats(stats, stats_path)
            logger.info(
                "Async checkpoint after batch %s: %s rows | est. cost $%.4f",
                batch_idx,
                len(rows),
                cost,
            )
            if cost >= cost_stop_usd:
                stats["cost_stop"] = 1
                stop_event.set()
                logger.error("Cost safety stop at $%.4f (limit $%.2f).", cost, cost_stop_usd)

    queue: asyncio.Queue[tuple[int, list[tuple[str, pd.Series]]] | None] = asyncio.Queue()
    for item in enumerate(batches, start=1):
        await queue.put(item)

    async def worker() -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                if stop_event.is_set():
                    continue
                batch_idx, batch = item
                await process_batch(batch_idx, batch)
            finally:
                queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    try:
        await queue.join()
    finally:
        for _ in workers:
            queue.put_nowait(None)
        await asyncio.gather(*workers, return_exceptions=True)
        await client.close()
    result = pd.DataFrame(rows)
    stats["tested"] = len(result)
    stats["reviews_processed_this_run"] = max(0, len(result) - prior_count)
    _refresh_cost(stats)
    save_checkpoint(rows, output_path)
    save_stats(stats, stats_path)
    return result, stats

def run(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
    limit: int | None = CLASSIFY_PILOT_LIMIT,
    random_sample: bool = False,
    seed: int = CLASSIFY_PILOT_SEED,
    model_override: str | None = None,
    sleep_seconds: float = CLASSIFY_SLEEP_SECONDS,
    resume: bool = False,
    batch_size: int = 1,
    stats_path: Path | None = None,
    skip_keys: set[str] | None = None,
    pace_for_tpm: bool = False,
    unclassified_limit: int | None = None,
    provider_override: str | None = None,
    tpm_limit: int = CLASSIFY_TPM_LIMIT,
    cost_stop_usd: float | None = None,
    max_prompt_tokens: int = CLASSIFY_MAX_PROMPT_TOKENS_PER_BATCH,
) -> tuple[Path, pd.DataFrame, dict[str, Any], str, str]:
    client, provider, model = build_chat_client(
        provider=provider_override,
        model=model_override,
    )
    reviews = load_reviews(
        input_path,
        limit=limit,
        random_sample=random_sample,
        seed=seed,
    )
    if skip_keys:
        keep = [review_key(row) not in skip_keys for _, row in reviews.iterrows()]
        reviews = reviews.loc[keep].reset_index(drop=True)
    if unclassified_limit is not None:
        reviews = reviews.head(unclassified_limit).reset_index(drop=True)
    existing = load_checkpoint(output_path) if resume else pd.DataFrame()
    logger.info(
        "Loaded %s reviews (limit=%s, random_sample=%s, resume=%s, batch_size=%s, provider=%s, model=%s)",
        len(reviews),
        limit,
        random_sample,
        resume,
        batch_size,
        provider,
        model,
    )
    analyzed, stats = analyze_dataframe(
        reviews,
        client,
        model=model,
        output_path=output_path if resume or limit is None or unclassified_limit is not None else None,
        sleep_seconds=sleep_seconds,
        existing=existing if resume else None,
        stats_path=stats_path,
        batch_size=batch_size,
        pace_for_tpm=pace_for_tpm,
        tpm_limit=tpm_limit,
        cost_stop_usd=cost_stop_usd,
        max_prompt_tokens=max_prompt_tokens,
    )
    out = save_analyzed(analyzed, output_path)
    return out, analyzed, stats, provider, model

def print_benchmark_report(
    df: pd.DataFrame,
    stats: dict[str, Any],
    output: Path,
    provider: str,
    model: str,
    duration_seconds: float,
    remaining_reviews: int,
) -> None:
    processed = int(
        stats.get("reviews_processed_this_run")
        or (stats.get("success", 0) + stats.get("failed", 0) - stats.get("resumed", 0))
    )
    api_requests = int(stats.get("api_requests") or 0)
    rpm = (processed / duration_seconds * 60.0) if duration_seconds > 0 else 0.0
    avg_per_req = (processed / api_requests) if api_requests else 0.0
    eta_min = (remaining_reviews / rpm) if rpm > 0 else float("inf")
    this_success = processed - int(stats.get("failed", 0))
    if this_success < 0:
        this_success = int(stats.get("success", 0))
    pydantic_ok = int(stats.get("pydantic_failures") or 0) == 0
    print("=== Flavor Scout Step 3 BATCH BENCHMARK (100 reviews) ===")
    print(f"LLM provider:                 {provider}")
    print(f"Model:                        {model}")
    print(f"Batch size used:              {stats.get('batch_size_configured')}")
    print(f"API requests:                 {api_requests}")
    print(f"Reviews processed:            {processed}")
    print(f"Average reviews per request:  {avg_per_req:.2f}")
    print(f"Total processing time (sec):  {duration_seconds:.1f}")
    print(f"Reviews per minute:           {rpm:.2f}")
    print(f"HTTP 429 count:               {stats.get('http_429', 0)}")
    print(f"Retries:                      {stats.get('retries', 0)}")
    print(f"Successfully classified:      {this_success}")
    print(f"Failed classifications:       {stats.get('failed', 0)}")
    print(f"Pydantic passed:              {stats.get('pydantic_passed', 0)}")
    print(f"Pydantic failures:            {stats.get('pydantic_failures', 0)}")
    print(f"All stored rows Pydantic-ok:  {pydantic_ok and int(stats.get('failed', 0)) == 0}")
    print(f"Checkpoint prior rows:        {stats.get('resumed', 0)}")
    print(f"Estimated minutes remaining:  {eta_min:.1f} for {remaining_reviews} reviews")
    print(f"Saved to:                     {output}")

def print_openai_benchmark_report(
    df: pd.DataFrame,
    stats: dict[str, Any],
    output: Path,
    provider: str,
    model: str,
    duration_seconds: float,
    remaining_reviews: int,
) -> None:
    processed = int(
        stats.get("reviews_processed_this_run")
        or (stats.get("success", 0) + stats.get("failed", 0) - stats.get("resumed", 0))
    )
    api_requests = int(stats.get("api_requests") or 0)
    rpm = (processed / duration_seconds * 60.0) if duration_seconds > 0 else 0.0
    avg_per_req = (processed / api_requests) if api_requests else 0.0
    latency_n = int(stats.get("request_latency_count") or 0)
    avg_latency = (
        float(stats.get("request_latency_sum") or 0.0) / latency_n if latency_n else 0.0
    )
    prompt_tokens = int(stats.get("prompt_tokens") or 0)
    completion_tokens = int(stats.get("completion_tokens") or 0)
    total_tokens = int(stats.get("tokens_used") or (prompt_tokens + completion_tokens))
    cost = estimate_openai_cost(prompt_tokens, completion_tokens)
    cost_per_review = (cost / processed) if processed else 0.0
    projected = cost_per_review * remaining_reviews
    this_success = int(stats.get("success", 0)) - 0
    pydantic_ok = int(stats.get("pydantic_failures") or 0) == 0 and int(stats.get("failed", 0)) == 0
    print("=== Flavor Scout Step 3 OPENAI COST BENCHMARK (100 reviews) ===")
    print(f"LLM provider:                 {provider}")
    print(f"Model:                        {model}")
    print(f"Batch size used:              {stats.get('batch_size_configured')}")
    print(f"Reviews processed:            {processed}")
    print(f"Successfully classified:      {this_success}")
    print(f"Failed classifications:       {stats.get('failed', 0)}")
    print(f"API requests:                 {api_requests}")
    print(f"Average reviews per request:  {avg_per_req:.2f}")
    print(f"Input tokens:                 {prompt_tokens}")
    print(f"Output tokens:                {completion_tokens}")
    print(f"Total tokens:                 {total_tokens}")
    print(f"Benchmark cost (USD):         ${cost:.4f}")
    print(f"Average cost/review (USD):    ${cost_per_review:.6f}")
    print(f"Projected remaining cost:     ${projected:.2f} for {remaining_reviews} reviews")
    print(f"$5 credit sufficient?:        {'yes' if projected <= 5.0 else 'no'}")
    print(f"Total processing time (sec):  {duration_seconds:.1f}")
    print(f"Reviews per minute:           {rpm:.2f}")
    print(f"Average request latency (s):  {avg_latency:.2f}")
    print(f"HTTP 429 count:               {stats.get('http_429', 0)}")
    print(f"Retries:                      {stats.get('retries', 0)}")
    print(f"Pydantic passed:              {stats.get('pydantic_passed', 0)}")
    print(f"Pydantic failures:            {stats.get('pydantic_failures', 0)}")
    print(f"All stored rows Pydantic-ok:  {pydantic_ok}")
    print(f"Saved to:                     {output}")

def print_quality_sample(df: pd.DataFrame, n: int = 8) -> None:
    classified = df[df["classification_status"] == "success"] if "classification_status" in df.columns else df
    if classified.empty:
        print("Quality check: no successful rows to inspect.")
        return
    sample = classified.head(n)
    print("=== Quality sample (first rows; check IDs are not mixed) ===")
    required = ["relevant", "flavor", "sentiment", "intent", "pain_point", "brand_fit", "reasoning", "confidence"]
    for _, row in sample.iterrows():
        text = str(row.get("review_text") or "").replace("\n", " ")[:90]
        missing = [c for c in required if c not in row or pd.isna(row.get(c)) and c not in {"flavor", "pain_point"}]
        print(
            f"- key={row.get('review_key')} | relevant={row.get('relevant')} | "
            f"flavor={row.get('flavor')} | intent={row.get('intent')} | "
            f"sentiment={row.get('sentiment')} | brand_fit={row.get('brand_fit')} | "
            f"confidence={row.get('confidence')}"
        )
        print(f"  review: {text}")
        print(f"  reasoning: {str(row.get('reasoning') or '')[:160]}")
        if missing:
            print(f"  MISSING FIELDS: {missing}")
    mismatched = 0
    for _, row in classified.iterrows():
        expected = review_key(row)
        if str(row.get("review_key")) != expected:
            mismatched += 1
    print(f"review_key mismatches vs source fields: {mismatched}")
    empty_reason = int(classified["reasoning"].fillna("").astype(str).str.strip().eq("").sum()) if "reasoning" in classified.columns else -1
    print(f"Empty reasoning rows: {empty_reason}")
    keys = classified["review_key"].astype(str) if "review_key" in classified.columns else pd.Series(dtype=str)
    print(f"Unique review_keys: {keys.nunique()} / {len(keys)} rows")
    print(f"Duplicate review_keys: {int(keys.duplicated().sum()) if len(keys) else 0}")
    flavors = classified["flavor"].fillna("").astype(str).str.strip() if "flavor" in classified.columns else pd.Series(dtype=str)
    flavors = flavors[flavors.ne("") & flavors.str.lower().ne("nan") & flavors.str.lower().ne("none")]
    banned = flavors[flavors.map(lambda x: _is_non_launchable_flavor(str(x)))]
    print(f"Launchable flavor mentions: {len(flavors)}")
    print(f"Banned flavor hits: {len(banned)} {sorted(set(banned.str.lower().tolist()))}")

def _checkpoint_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    existing = pd.read_csv(path)
    keys: set[str] = set()
    for _, row in existing.iterrows():
        keys.add(str(row.get("review_key") or review_key(row)))
    return keys

def main() -> None:
    parser = argparse.ArgumentParser(description="Flavor Scout Step 3 LLM classification")
    parser.add_argument("--validation-v2", action="store_true")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Classify all processed reviews into analyzed_reviews.csv (resumable).",
    )
    parser.add_argument(
        "--batch-benchmark",
        action="store_true",
        help="Classify 100 reviews after the existing full checkpoint into a separate test CSV.",
    )
    parser.add_argument(
        "--openai-benchmark",
        action="store_true",
        help="Classify 100 unclassified reviews with gpt-4.1-mini into openai_100_benchmark.csv.",
    )
    parser.add_argument(
        "--openai-batch8-benchmark",
        action="store_true",
        help="Classify 100 unclassified reviews with gpt-4.1-mini using batch size 8.",
    )
    parser.add_argument(
        "--openai-quality-validation",
        action="store_true",
        help="100-review quality validation after flavor/relevance prompt fixes (batch size 8).",
    )
    parser.add_argument(
        "--openai-final-quality-validation",
        action="store_true",
        help="Final 100-review SKU-based relevance validation (batch size 8).",
    )
    parser.add_argument(
        "--openai-async-benchmark",
        action="store_true",
        help="100-review async benchmark: batch size 8, max 3 concurrent OpenAI requests.",
    )
    args = parser.parse_args()

    if sum(
        [
            args.full,
            args.validation_v2,
            args.batch_benchmark,
            args.openai_benchmark,
            args.openai_batch8_benchmark,
            args.openai_quality_validation,
            args.openai_final_quality_validation,
            args.openai_async_benchmark,
        ]
    ) > 1:
        raise SystemExit("Use only one classification mode flag at a time.")

    if args.openai_async_benchmark:
        output_path = OUTPUT_PATH_OPENAI_ASYNC
        if output_path.resolve() in EXISTING_CLASSIFIED_OUTPUTS:
            raise SystemExit("Refusing to overwrite an existing classified file.")
        started = time.time()
        skip_keys = (
            _checkpoint_keys(OUTPUT_PATH_FULL)
            | _checkpoint_keys(OUTPUT_PATH_BATCH_TEST)
            | _checkpoint_keys(OUTPUT_PATH_OPENAI_BENCH)
            | _checkpoint_keys(OUTPUT_PATH_OPENAI_BATCH8)
            | _checkpoint_keys(OUTPUT_PATH_OPENAI_QUALITY)
            | _checkpoint_keys(OUTPUT_PATH_OPENAI_FINAL_QUALITY)
        )
        reviews = load_reviews(INPUT_PATH, limit=None, random_sample=False)
        keep = [review_key(row) not in skip_keys for _, row in reviews.iterrows()]
        reviews = reviews.loc[keep].head(CLASSIFY_BATCH_BENCHMARK_LIMIT).reset_index(drop=True)
        logger.info(
            "Async benchmark: %s unclassified reviews, batch_size=8, concurrency=%s, model=%s",
            len(reviews),
            CLASSIFY_ASYNC_CONCURRENCY,
            OPENAI_MODEL_BENCHMARK,
        )
        try:
            analyzed, stats = asyncio.run(
                run_async_openai_benchmark(
                    reviews=reviews,
                    output_path=output_path,
                    stats_path=STATS_PATH_OPENAI_ASYNC,
                    model=OPENAI_MODEL_BENCHMARK,
                    batch_size=CLASSIFY_OPENAI_BATCH_SIZE_TEST,
                    max_prompt_tokens=CLASSIFY_OPENAI_MAX_PROMPT_TOKENS_PER_BATCH,
                    concurrency=CLASSIFY_ASYNC_CONCURRENCY,
                    cost_stop_usd=CLASSIFY_COST_STOP_USD,
                )
            )
            duration = time.time() - started
            processed = int(stats.get("reviews_processed_this_run") or len(analyzed))
            cost = estimate_openai_cost(
                int(stats.get("prompt_tokens") or 0),
                int(stats.get("completion_tokens") or 0),
            )
            stats["duration_seconds"] = duration
            stats["benchmark_cost_usd"] = cost
            stats["cost_per_review_usd"] = (cost / processed) if processed else 0.0
            save_analyzed(analyzed, output_path)
            save_stats(stats, STATS_PATH_OPENAI_ASYNC)
            print_openai_benchmark_report(
                analyzed, stats, output_path, "openai", OPENAI_MODEL_BENCHMARK, duration, 7267
            )
            print(f"Concurrent workers:           {CLASSIFY_ASYNC_CONCURRENCY}")
            print(f"SKU relevance overrides:      {stats.get('sku_relevance_overrides', 0)}")
            print_quality_sample(analyzed)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("%s", exc)
            sys.exit(1)
        except Exception as exc:
            logger.error("Classification failed: %s", exc)
            sys.exit(1)
        return

    if args.openai_final_quality_validation:
        output_path = OUTPUT_PATH_OPENAI_FINAL_QUALITY
        if output_path.resolve() in EXISTING_CLASSIFIED_OUTPUTS:
            raise SystemExit("Refusing to overwrite an existing classified file.")
        started = time.time()
        skip_keys = (
            _checkpoint_keys(OUTPUT_PATH_FULL)
            | _checkpoint_keys(OUTPUT_PATH_BATCH_TEST)
            | _checkpoint_keys(OUTPUT_PATH_OPENAI_BENCH)
            | _checkpoint_keys(OUTPUT_PATH_OPENAI_BATCH8)
            | _checkpoint_keys(OUTPUT_PATH_OPENAI_QUALITY)
        )
        try:
            output, analyzed, stats, provider, model = run(
                output_path=output_path,
                limit=None,
                random_sample=False,
                model_override=OPENAI_MODEL_BENCHMARK,
                sleep_seconds=CLASSIFY_SLEEP_SECONDS,
                resume=True,
                batch_size=CLASSIFY_OPENAI_BATCH_SIZE_TEST,
                stats_path=STATS_PATH_OPENAI_FINAL_QUALITY,
                skip_keys=skip_keys,
                pace_for_tpm=True,
                unclassified_limit=CLASSIFY_BATCH_BENCHMARK_LIMIT,
                provider_override="openai",
                tpm_limit=OPENAI_TPM_LIMIT,
                cost_stop_usd=CLASSIFY_COST_STOP_USD,
                max_prompt_tokens=CLASSIFY_OPENAI_MAX_PROMPT_TOKENS_PER_BATCH,
            )
            duration = time.time() - started
            processed = int(stats.get("reviews_processed_this_run") or len(analyzed))
            cost = estimate_openai_cost(
                int(stats.get("prompt_tokens") or 0),
                int(stats.get("completion_tokens") or 0),
            )
            stats["duration_seconds"] = duration
            stats["benchmark_cost_usd"] = cost
            stats["cost_per_review_usd"] = (cost / processed) if processed else 0.0
            save_stats(stats, STATS_PATH_OPENAI_FINAL_QUALITY)
            print_openai_benchmark_report(
                analyzed, stats, output, provider, model, duration, 7267
            )
            print_quality_sample(analyzed)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("%s", exc)
            sys.exit(1)
        except Exception as exc:
            logger.error("Classification failed: %s", exc)
            sys.exit(1)
        return

    if args.openai_quality_validation:
        output_path = OUTPUT_PATH_OPENAI_QUALITY
        if output_path.resolve() in EXISTING_CLASSIFIED_OUTPUTS:
            raise SystemExit("Refusing to overwrite an existing classified file.")
        started = time.time()
        skip_keys = (
            _checkpoint_keys(OUTPUT_PATH_FULL)
            | _checkpoint_keys(OUTPUT_PATH_BATCH_TEST)
            | _checkpoint_keys(OUTPUT_PATH_OPENAI_BENCH)
            | _checkpoint_keys(OUTPUT_PATH_OPENAI_BATCH8)
        )
        try:
            output, analyzed, stats, provider, model = run(
                output_path=output_path,
                limit=None,
                random_sample=False,
                model_override=OPENAI_MODEL_BENCHMARK,
                sleep_seconds=CLASSIFY_SLEEP_SECONDS,
                resume=True,
                batch_size=CLASSIFY_OPENAI_BATCH_SIZE_TEST,
                stats_path=STATS_PATH_OPENAI_QUALITY,
                skip_keys=skip_keys,
                pace_for_tpm=True,
                unclassified_limit=CLASSIFY_BATCH_BENCHMARK_LIMIT,
                provider_override="openai",
                tpm_limit=OPENAI_TPM_LIMIT,
                cost_stop_usd=CLASSIFY_COST_STOP_USD,
                max_prompt_tokens=CLASSIFY_OPENAI_MAX_PROMPT_TOKENS_PER_BATCH,
            )
            duration = time.time() - started
            processed = int(stats.get("reviews_processed_this_run") or len(analyzed))
            prompt_tokens = int(stats.get("prompt_tokens") or 0)
            completion_tokens = int(stats.get("completion_tokens") or 0)
            cost = estimate_openai_cost(prompt_tokens, completion_tokens)
            stats["duration_seconds"] = duration
            stats["benchmark_cost_usd"] = cost
            stats["cost_per_review_usd"] = (cost / processed) if processed else 0.0
            save_stats(stats, STATS_PATH_OPENAI_QUALITY)
            print_openai_benchmark_report(
                analyzed, stats, output, provider, model, duration, 7267
            )
            print_quality_sample(analyzed)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("%s", exc)
            sys.exit(1)
        except Exception as exc:
            logger.error("Classification failed: %s", exc)
            sys.exit(1)
        return

    if args.openai_batch8_benchmark:
        output_path = OUTPUT_PATH_OPENAI_BATCH8
        if output_path.resolve() in EXISTING_CLASSIFIED_OUTPUTS:
            raise SystemExit("Refusing to overwrite an existing classified file.")
        started = time.time()
        skip_keys = (
            _checkpoint_keys(OUTPUT_PATH_FULL)
            | _checkpoint_keys(OUTPUT_PATH_BATCH_TEST)
            | _checkpoint_keys(OUTPUT_PATH_OPENAI_BENCH)
        )
        n_processed = len(load_reviews(limit=None))
        remaining_unclassified = max(0, n_processed - len(skip_keys))
        try:
            output, analyzed, stats, provider, model = run(
                output_path=output_path,
                limit=None,
                random_sample=False,
                model_override=OPENAI_MODEL_BENCHMARK,
                sleep_seconds=CLASSIFY_SLEEP_SECONDS,
                resume=True,
                batch_size=CLASSIFY_OPENAI_BATCH_SIZE_TEST,
                stats_path=STATS_PATH_OPENAI_BATCH8,
                skip_keys=skip_keys,
                pace_for_tpm=True,
                unclassified_limit=CLASSIFY_BATCH_BENCHMARK_LIMIT,
                provider_override="openai",
                tpm_limit=OPENAI_TPM_LIMIT,
                cost_stop_usd=CLASSIFY_COST_STOP_USD,
                max_prompt_tokens=CLASSIFY_OPENAI_MAX_PROMPT_TOKENS_PER_BATCH,
            )
            duration = time.time() - started
            processed = int(stats.get("reviews_processed_this_run") or len(analyzed))
            prompt_tokens = int(stats.get("prompt_tokens") or 0)
            completion_tokens = int(stats.get("completion_tokens") or 0)
            cost = estimate_openai_cost(prompt_tokens, completion_tokens)
            cost_per_review = (cost / processed) if processed else 0.0
            still_remaining = 7267
            stats["duration_seconds"] = duration
            stats["benchmark_cost_usd"] = cost
            stats["cost_per_review_usd"] = cost_per_review
            stats["projected_remaining_cost_usd"] = cost_per_review * still_remaining
            save_stats(stats, STATS_PATH_OPENAI_BATCH8)
            print_openai_benchmark_report(
                analyzed, stats, output, provider, model, duration, still_remaining
            )
            print_quality_sample(analyzed)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("%s", exc)
            sys.exit(1)
        except Exception as exc:
            logger.error("Classification failed: %s", exc)
            sys.exit(1)
        return

    if args.openai_benchmark:
        output_path = OUTPUT_PATH_OPENAI_BENCH
        if output_path.resolve() in EXISTING_CLASSIFIED_OUTPUTS:
            raise SystemExit("Refusing to overwrite an existing classified file.")
        started = time.time()
        skip_keys = _checkpoint_keys(OUTPUT_PATH_FULL) | _checkpoint_keys(OUTPUT_PATH_BATCH_TEST)
        n_processed = len(load_reviews(limit=None))
        remaining_unclassified = max(0, n_processed - len(skip_keys))
        try:
            output, analyzed, stats, provider, model = run(
                output_path=output_path,
                limit=None,
                random_sample=False,
                model_override=OPENAI_MODEL_BENCHMARK,
                sleep_seconds=CLASSIFY_SLEEP_SECONDS,
                resume=True,
                batch_size=CLASSIFY_BATCH_SIZE,
                stats_path=STATS_PATH_OPENAI_BENCH,
                skip_keys=skip_keys,
                pace_for_tpm=True,
                unclassified_limit=CLASSIFY_BATCH_BENCHMARK_LIMIT,
                provider_override="openai",
                tpm_limit=OPENAI_TPM_LIMIT,
            )
            duration = time.time() - started
            processed = int(stats.get("reviews_processed_this_run") or len(analyzed))
            prompt_tokens = int(stats.get("prompt_tokens") or 0)
            completion_tokens = int(stats.get("completion_tokens") or 0)
            cost = estimate_openai_cost(prompt_tokens, completion_tokens)
            cost_per_review = (cost / processed) if processed else 0.0
            still_remaining = max(0, remaining_unclassified - processed)
            stats["duration_seconds"] = duration
            stats["benchmark_cost_usd"] = cost
            stats["cost_per_review_usd"] = cost_per_review
            stats["projected_remaining_cost_usd"] = cost_per_review * still_remaining
            stats["five_dollar_sufficient"] = bool(stats["projected_remaining_cost_usd"] <= 5.0)
            save_stats(stats, STATS_PATH_OPENAI_BENCH)
            print_openai_benchmark_report(
                analyzed, stats, output, provider, model, duration, still_remaining
            )
            print_quality_sample(analyzed)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("%s", exc)
            sys.exit(1)
        except Exception as exc:
            logger.error("Classification failed: %s", exc)
            sys.exit(1)
        return

    if args.batch_benchmark:
        if OUTPUT_PATH_BATCH_TEST.resolve() in PROTECTED_OUTPUTS:
            raise SystemExit("Refusing to overwrite a pilot file.")
        started = time.time()
        skip_keys = _checkpoint_keys(OUTPUT_PATH_FULL)
        n_processed = len(load_reviews(limit=None))
        remaining_after_full = n_processed - len(skip_keys)
        try:
            output, analyzed, stats, provider, model = run(
                output_path=OUTPUT_PATH_BATCH_TEST,
                limit=None,
                random_sample=False,
                model_override=GROQ_MODEL_FULL,
                sleep_seconds=CLASSIFY_FULL_SLEEP_SECONDS,
                resume=True,
                batch_size=CLASSIFY_BATCH_SIZE,
                stats_path=STATS_PATH_BATCH_TEST,
                skip_keys=skip_keys,
                pace_for_tpm=True,
                unclassified_limit=CLASSIFY_BATCH_BENCHMARK_LIMIT,
            )
            duration = time.time() - started
            processed = int(stats.get("reviews_processed_this_run") or len(analyzed))
            rpm = (processed / duration * 60.0) if duration > 0 else 0.0
            still_remaining = max(0, remaining_after_full - processed)
            stats["duration_seconds"] = duration
            stats["reviews_per_minute"] = rpm
            stats["estimated_minutes_remaining"] = (still_remaining / rpm) if rpm else None
            save_stats(stats, STATS_PATH_BATCH_TEST)
            print_benchmark_report(
                analyzed, stats, output, provider, model, duration, still_remaining
            )
            print_quality_sample(analyzed)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("%s", exc)
            sys.exit(1)
        except Exception as exc:
            logger.error("Classification failed: %s", exc)
            sys.exit(1)
        return

    if args.full:
        output_path = OUTPUT_PATH_FULL
        if output_path.resolve() in PROTECTED_OUTPUTS:
            raise SystemExit("Refusing to overwrite a pilot file.")
        started = time.time()
        try:
            existing = load_checkpoint(output_path)
            all_reviews = load_reviews(INPUT_PATH, limit=None, random_sample=False)
            skip_keys = {
                str(row.get("review_key") or review_key(row))
                for _, row in existing.iterrows()
                if str(row.get("classification_status") or "") == "success"
            }
            keep = [review_key(row) not in skip_keys for _, row in all_reviews.iterrows()]
            pending = all_reviews.loc[keep].reset_index(drop=True)
            logger.info(
                "Full async run: resume %s success keys, pending %s, batch=8, concurrency=%s",
                len(skip_keys),
                len(pending),
                CLASSIFY_ASYNC_CONCURRENCY,
            )
            analyzed, stats = asyncio.run(
                run_async_openai_benchmark(
                    reviews=pending,
                    output_path=output_path,
                    stats_path=STATS_PATH_FULL,
                    model=OPENAI_MODEL_BENCHMARK,
                    batch_size=CLASSIFY_OPENAI_BATCH_SIZE_TEST,
                    max_prompt_tokens=CLASSIFY_OPENAI_MAX_PROMPT_TOKENS_PER_BATCH,
                    concurrency=CLASSIFY_ASYNC_CONCURRENCY,
                    cost_stop_usd=CLASSIFY_COST_STOP_USD,
                    existing=existing,
                )
            )
            duration = time.time() - started
            stats["duration_seconds"] = duration
            _refresh_cost(stats)
            save_analyzed(analyzed, output_path)
            save_stats(stats, STATS_PATH_FULL)
            print_full_summary(analyzed, stats, output_path, "openai", OPENAI_MODEL_BENCHMARK, duration)
            print(f"Concurrent workers:           {CLASSIFY_ASYNC_CONCURRENCY}")
            print(f"Newly classified this run:    {stats.get('reviews_processed_this_run')}")
            print(f"SKU relevance overrides:      {stats.get('sku_relevance_overrides', 0)}")
            print_quality_sample(analyzed, n=8)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("%s", exc)
            sys.exit(1)
        except Exception as exc:
            logger.error("Classification failed: %s", exc)
            sys.exit(1)
        return

    output_path = OUTPUT_PATH_V2 if args.validation_v2 else OUTPUT_PATH
    try:
        output, analyzed, stats, provider, model = run(
            output_path=output_path,
            random_sample=bool(args.validation_v2),
        )
        print_pilot_summary(analyzed, stats, output, provider, model)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("Classification failed: %s", exc)
        sys.exit(1)

if __name__ == "__main__":
    main()
