"""
Step 4: Deterministic Trend Engine.

Reads:  data/processed/analyzed_reviews.csv  (unchanged)
Writes: data/processed/flavor_mention_qc.csv
        data/processed/flavor_trends.csv
        data/processed/flavor_trends_stats.json

Does NOT call an LLM. Does NOT compute Opportunity Score.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.aggregation.flavor_qc import aggregation_flavor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "analyzed_reviews.csv"
MENTION_QC_PATH = PROJECT_ROOT / "data" / "processed" / "flavor_mention_qc.csv"
TRENDS_PATH = PROJECT_ROOT / "data" / "processed" / "flavor_trends.csv"
STATS_PATH = PROJECT_ROOT / "data" / "processed" / "flavor_trends_stats.json"

MIN_MENTIONS_FOR_SCORING = 2
MIN_MENTIONS_FOR_GROWTH = 5
MIN_WINDOW_MENTIONS_FOR_GROWTH = 2

BRAND_FIT_POINTS = {"strong": 1.0, "moderate": 2 / 3, "weak": 1 / 3, "none": 0.0}
DEMAND_INTENTS = {"purchase_intent", "request"}


def _is_true(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _intent(value: object) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def _sentiment(value: object) -> str:
    key = str(value or "").strip().lower()
    if key in {"positive", "negative", "neutral"}:
        return key
    return "other"


def _brand_fit(value: object) -> str:
    key = str(value or "").strip().lower()
    return key if key in BRAND_FIT_POINTS else "none"


def _confidence(value: object) -> str:
    key = str(value or "").strip().lower()
    return key if key in {"high", "medium", "low"} else "unknown"


def load_classified(path: Path = INPUT_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing classified reviews: {path}")
    df = pd.read_csv(path)
    if len(df) != 7931:
        logger.warning("Expected 7,931 classified rows; found %s", len(df))
    return df


def build_mention_qc(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, rec in df.iterrows():
        flavor, reason, merge = aggregation_flavor(rec.get("flavor"))
        relevant = _is_true(rec.get("relevant"))
        eligible = bool(relevant and flavor)
        if not relevant and flavor:
            reason = reason or "irrelevant_sku"
            flavor = None
            eligible = False
        rows.append(
            {
                "review_key": rec.get("review_key"),
                "relevant": relevant,
                "flavor": rec.get("flavor") if pd.notna(rec.get("flavor")) else "",
                "aggregation_flavor": flavor or "",
                "eligible": eligible,
                "exclusion_reason": "" if eligible else (reason or "not_relevant_or_no_flavor"),
                "normalization_merge": merge or "",
                "sentiment": _sentiment(rec.get("sentiment")),
                "intent": _intent(rec.get("intent")),
                "brand_fit": _brand_fit(rec.get("brand_fit")),
                "confidence": _confidence(rec.get("confidence")),
                "timestamp": rec.get("timestamp"),
            }
        )
    return pd.DataFrame(rows)


def _growth_split(timestamps: pd.Series) -> tuple[pd.Timestamp | None, str]:
    parsed = pd.to_datetime(timestamps, utc=True, errors="coerce")
    valid = parsed.dropna()
    if len(valid) < MIN_MENTIONS_FOR_GROWTH:
        return None, "insufficient_eligible_volume"
    span_days = (valid.max() - valid.min()).days
    if span_days < 180:
        return None, "insufficient_time_span"
    midpoint = valid.min() + (valid.max() - valid.min()) / 2
    return midpoint, "midpoint_of_eligible_window"


def aggregate_trends(mentions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    eligible = mentions[mentions["eligible"]].copy()
    eligible["ts"] = pd.to_datetime(eligible["timestamp"], utc=True, errors="coerce")
    midpoint, growth_method = _growth_split(eligible["timestamp"])

    records: list[dict[str, Any]] = []
    for flavor, grp in eligible.groupby("aggregation_flavor", dropna=False):
        n = len(grp)
        sent = grp["sentiment"]
        pos = int((sent == "positive").sum())
        neg = int((sent == "negative").sum())
        neu = int((sent == "neutral").sum())
        intents = grp["intent"]
        purchase = int((intents == "purchase_intent").sum())
        request = int((intents == "request").sum())
        praise = int((intents == "praise").sum())
        complaint = int((intents == "complaint").sum())
        preference = int((intents == "preference").sum())
        comparison = int((intents == "comparison").sum())
        general = int((intents == "general_mention").sum())
        fit = grp["brand_fit"]
        brand_score = float(fit.map(lambda x: BRAND_FIT_POINTS.get(x, 0.0)).mean()) if n else 0.0
        conf = grp["confidence"].value_counts()
        conf_mode = str(conf.index[0]) if len(conf) else "unknown"
        high_share = float((grp["confidence"] == "high").mean()) if n else 0.0

        growth: float | None = None
        early = late = 0
        if midpoint is not None and n >= MIN_MENTIONS_FOR_GROWTH:
            known = grp["ts"].notna()
            early = int((known & (grp["ts"] < midpoint)).sum())
            late = int((known & (grp["ts"] >= midpoint)).sum())
            if early >= MIN_WINDOW_MENTIONS_FOR_GROWTH and late >= MIN_WINDOW_MENTIONS_FOR_GROWTH:
                growth = (late - early) / early
            else:
                growth = None

        records.append(
            {
                "flavor": flavor,
                "mentions": n,
                "demand": n,
                "positive_count": pos,
                "negative_count": neg,
                "neutral_count": neu,
                "positive_rate": round(pos / n, 4) if n else 0.0,
                "negative_rate": round(neg / n, 4) if n else 0.0,
                "neutral_rate": round(neu / n, 4) if n else 0.0,
                "purchase_intent_count": purchase,
                "request_count": request,
                "demand_intent_count": purchase + request,
                "praise_count": praise,
                "complaint_count": complaint,
                "preference_count": preference,
                "comparison_count": comparison,
                "general_mention_count": general,
                "brand_fit_score": round(brand_score, 4),
                "brand_fit_strong": int((fit == "strong").sum()),
                "brand_fit_moderate": int((fit == "moderate").sum()),
                "brand_fit_weak": int((fit == "weak").sum()),
                "brand_fit_none": int((fit == "none").sum()),
                "confidence_mode": conf_mode,
                "confidence_high_rate": round(high_share, 4),
                "growth": growth,
                "growth_early_mentions": early if midpoint is not None else None,
                "growth_late_mentions": late if midpoint is not None else None,
                "eligible_for_scoring": n >= MIN_MENTIONS_FOR_SCORING,
            }
        )

    trends = pd.DataFrame(records)
    if not trends.empty:
        trends = trends.sort_values(
            ["mentions", "demand_intent_count", "positive_rate"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    stats: dict[str, Any] = {
        "source_reviews": 7931,
        "classified_reviews": int(len(mentions)),
        "relevant_reviews": int(mentions["relevant"].sum()),
        "irrelevant_reviews": int((~mentions["relevant"]).sum()),
        "raw_nonzero_flavor_mentions": int(
            mentions["flavor"].fillna("").astype(str).str.strip().replace({"nan": ""}).ne("").sum()
        ),
        "eligible_flavor_mentions": int(eligible.shape[0]),
        "unique_eligible_flavors": int(trends.shape[0]),
        "excluded_meal_or_dish": int((mentions["exclusion_reason"] == "meal_or_dish").sum()),
        "excluded_generic_non_launchable": int(
            (mentions["exclusion_reason"] == "generic_non_launchable").sum()
        ),
        "normalization_merges": int(mentions["normalization_merge"].fillna("").astype(str).ne("").sum()),
        "growth_method": growth_method,
        "growth_midpoint_utc": midpoint.isoformat() if midpoint is not None else None,
        "min_mentions_for_scoring": MIN_MENTIONS_FOR_SCORING,
        "flavors_eligible_for_scoring": int(trends["eligible_for_scoring"].sum()) if not trends.empty else 0,
        "flavors_with_numeric_growth": int(trends["growth"].notna().sum()) if not trends.empty else 0,
    }
    return trends, stats


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run(
    input_path: Path = INPUT_PATH,
    mention_path: Path = MENTION_QC_PATH,
    trends_path: Path = TRENDS_PATH,
    stats_path: Path = STATS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    classified = load_classified(input_path)
    mentions = build_mention_qc(classified)
    trends, stats = aggregate_trends(mentions)
    mention_path.parent.mkdir(parents=True, exist_ok=True)
    mentions.to_csv(mention_path, index=False)
    trends.to_csv(trends_path, index=False)
    save_json(stats, stats_path)
    logger.info("Wrote %s (%s rows)", mention_path, len(mentions))
    logger.info("Wrote %s (%s flavors)", trends_path, len(trends))
    logger.info("Wrote %s", stats_path)
    return mentions, trends, stats


def main() -> None:
    try:
        mentions, trends, stats = run()
    except Exception as exc:
        logger.error("%s", exc)
        sys.exit(1)
    print("=== Flavor Scout Step 4 TREND AGGREGATION ===")
    print(f"Classified reviews:           {stats['classified_reviews']}")
    print(f"Relevant:                     {stats['relevant_reviews']}")
    print(f"Irrelevant:                   {stats['irrelevant_reviews']}")
    print(f"Raw non-null flavor mentions: {stats['raw_nonzero_flavor_mentions']}")
    print(f"Eligible flavor mentions:     {stats['eligible_flavor_mentions']}")
    print(f"Unique eligible flavors:      {stats['unique_eligible_flavors']}")
    print(f"Excluded meal/dish:           {stats['excluded_meal_or_dish']}")
    print(f"Excluded generic:             {stats['excluded_generic_non_launchable']}")
    print(f"Normalization merges:         {stats['normalization_merges']}")
    print(f"Growth method:                {stats['growth_method']}")
    print(f"Numeric growth flavors:       {stats['flavors_with_numeric_growth']}")
    print(f"Eligible for later scoring:   {stats['flavors_eligible_for_scoring']}")
    print("Top flavors by demand:")
    cols = [
        "flavor",
        "mentions",
        "positive_rate",
        "demand_intent_count",
        "purchase_intent_count",
        "request_count",
        "brand_fit_score",
        "growth",
        "eligible_for_scoring",
    ]
    if not trends.empty:
        print(trends[cols].head(20).to_string(index=False))
    print(f"Mention QC: {MENTION_QC_PATH}")
    print(f"Trends:     {TRENDS_PATH}")
    print(f"Stats:      {STATS_PATH}")
    _ = mentions


if __name__ == "__main__":
    main()
