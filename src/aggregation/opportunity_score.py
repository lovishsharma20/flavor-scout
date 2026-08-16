"""
Step 5: Deterministic Opportunity Scoring.

Reads:  data/processed/flavor_trends.csv  (unchanged)
Writes: data/processed/opportunity_scores.csv
        data/processed/opportunity_scores_stats.json

Does NOT call an LLM. Does NOT assign a HealthKart brand.
Does NOT overwrite analyzed_reviews.csv, flavor_mention_qc.csv, or flavor_trends.csv.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRENDS_PATH = PROJECT_ROOT / "data" / "processed" / "flavor_trends.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "opportunity_scores.csv"
STATS_PATH = PROJECT_ROOT / "data" / "processed" / "opportunity_scores_stats.json"

# Assignment baseline: Demand 30, Growth 20, Sentiment 20, Purchase 15, Brand 15.
# Growth and Purchase Intent have no reliable signal in this dataset, so they
# are excluded from the numerical score and the remaining 65 points are renormalized.
WEIGHT_DEMAND = 30 / 65
WEIGHT_SENTIMENT = 20 / 65
WEIGHT_BRAND_FIT = 15 / 65

MEAL_OR_GENERIC = {
    "beef stroganoff",
    "sweet & sour pork",
    "sweet and sour pork",
    "rice and chicken",
    "pad thai",
    "cheesy hamburger",
    "cheese enchilada ranchero",
    "taco",
    "chicken",
    "shrimp",
    "beefish",
    "unflavored",
    "original",
    "sour",
    "fruit",
    "whey milk",
}


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def assign_confidence(mentions: int, positive_rate: float, brand_fit_score: float) -> tuple[str, str]:
    """
    Confidence is diagnostic, not part of Opportunity Score.

    HIGH:   >=5 mentions, positive_rate >= 0.80, brand_fit_score >= 0.50
    MEDIUM: >=3 mentions, or 2 mentions with positive_rate >= 0.75 and brand_fit_score >= 0.50
    LOW:    otherwise (thin or mixed evidence)

    Every reason notes that purchase intent and growth were unavailable.
    Missing-signal suffix is always appended.
    """
    missing = (
        "Purchase intent is 0 in this sample and temporal growth is not reliable, "
        "so those pillars were excluded from the score."
    )
    if mentions >= 5 and positive_rate >= 0.80 and brand_fit_score >= 0.50:
        level = "HIGH"
        core = (
            f"{mentions} mentions with {positive_rate:.1%} positive sentiment "
            f"and brand-fit {brand_fit_score:.2f}."
        )
    elif mentions >= 3 or (mentions >= 2 and positive_rate >= 0.75 and brand_fit_score >= 0.50):
        level = "MEDIUM"
        core = (
            f"{mentions} mentions; positive sentiment {positive_rate:.1%}; "
            f"brand-fit {brand_fit_score:.2f}. Volume is limited."
        )
    else:
        level = "LOW"
        core = (
            f"Only {mentions} mentions with positive sentiment {positive_rate:.1%} "
            f"and brand-fit {brand_fit_score:.2f}."
        )
    return level, f"{core} {missing}"


def score_trends(trends: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    eligible = trends[trends["eligible_for_scoring"].map(_bool)].copy()
    if len(eligible) != 11:
        raise ValueError(f"Expected 11 scoring-eligible flavors, found {len(eligible)}")
    if eligible["flavor"].duplicated().any():
        raise ValueError("Duplicate flavors in scoring set")
    if (eligible["mentions"] < 2).any():
        raise ValueError("Scoring set contains single-mention flavors")
    lowered = eligible["flavor"].astype(str).str.strip().str.lower()
    banned = lowered[lowered.isin(MEAL_OR_GENERIC)]
    if len(banned):
        raise ValueError(f"Non-launchable flavors in scoring set: {banned.tolist()}")

    min_m = float(eligible["mentions"].min())
    max_m = float(eligible["mentions"].max())
    span = max_m - min_m
    if span <= 0:
        raise ValueError("Cannot min-max normalize demand: mention span is 0")

    rows: list[dict[str, Any]] = []
    for _, rec in eligible.iterrows():
        mentions = int(rec["mentions"])
        demand_score = 100.0 * (mentions - min_m) / span
        positive_rate = float(rec["positive_rate"])
        sentiment_score = 100.0 * positive_rate
        brand_fit_0_1 = float(rec["brand_fit_score"])
        brand_fit_score = 100.0 * brand_fit_0_1
        opportunity = (
            WEIGHT_DEMAND * demand_score
            + WEIGHT_SENTIMENT * sentiment_score
            + WEIGHT_BRAND_FIT * brand_fit_score
        )
        confidence, reason = assign_confidence(mentions, positive_rate, brand_fit_0_1)
        growth_raw = rec.get("growth")
        growth_value = None if pd.isna(growth_raw) else float(growth_raw)
        rows.append(
            {
                "flavor": rec["flavor"],
                "mentions": mentions,
                "demand_score": round(demand_score, 4),
                "positive_rate": round(positive_rate, 4),
                "sentiment_score": round(sentiment_score, 4),
                "brand_fit_score": round(brand_fit_score, 4),
                "opportunity_score": round(opportunity, 4),
                "confidence": confidence,
                "confidence_reason": reason,
                "purchase_intent_count": int(rec.get("purchase_intent_count") or 0),
                "request_count": int(rec.get("request_count") or 0),
                "purchase_intent_available": False,
                "growth": growth_value,
                "growth_available": False,
                "eligible_for_scoring": True,
            }
        )

    scored = pd.DataFrame(rows).sort_values(
        ["opportunity_score", "mentions"],
        ascending=[False, False],
    ).reset_index(drop=True)
    scored.insert(0, "rank", range(1, len(scored) + 1))

    if scored["opportunity_score"].isna().any():
        raise ValueError("NaN opportunity scores")
    if ((scored["opportunity_score"] < 0) | (scored["opportunity_score"] > 100)).any():
        raise ValueError("Opportunity scores outside 0–100")
    for col in ("demand_score", "sentiment_score", "brand_fit_score"):
        if ((scored[col] < 0) | (scored[col] > 100)).any():
            raise ValueError(f"{col} outside 0–100")
    if scored["purchase_intent_count"].sum() != 0 or scored["request_count"].sum() != 0:
        logger.warning("Unexpected purchase/request counts in scoring set")
    if bool(scored["purchase_intent_available"].any()) or bool(scored["growth_available"].any()):
        raise ValueError("Purchase intent or growth marked available")

    stats = {
        "scoring_flavors": int(len(scored)),
        "mention_min": min_m,
        "mention_max": max_m,
        "weights": {
            "documented_baseline": {
                "demand": 0.30,
                "growth": 0.20,
                "sentiment": 0.20,
                "purchase_intent": 0.15,
                "brand_fit": 0.15,
            },
            "applied_after_renormalizing_available_65": {
                "demand": WEIGHT_DEMAND,
                "sentiment": WEIGHT_SENTIMENT,
                "brand_fit": WEIGHT_BRAND_FIT,
            },
            "excluded": ["growth", "purchase_intent"],
            "exclusion_reason": (
                "purchase_intent_count and request_count are 0 on all eligible mentions; "
                "temporal growth is not reliable at this sample size."
            ),
        },
        "top_flavor": scored.iloc[0]["flavor"],
        "top_score": float(scored.iloc[0]["opportunity_score"]),
    }
    return scored, stats


def validate(scored: pd.DataFrame) -> None:
    assert len(scored) == 11
    assert scored["flavor"].nunique() == 11
    assert scored["eligible_for_scoring"].all()
    assert (scored["mentions"] >= 2).all()
    assert scored["opportunity_score"].between(0, 100).all()
    assert not scored["purchase_intent_available"].any()
    assert not scored["growth_available"].any()
    assert not scored["opportunity_score"].isna().any()
    logger.info("Opportunity scoring validation passed (11 flavors).")


def run(
    trends_path: Path = TRENDS_PATH,
    output_path: Path = OUTPUT_PATH,
    stats_path: Path = STATS_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    trends = pd.read_csv(trends_path)
    scored, stats = score_trends(trends)
    validate(scored)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("Wrote %s", output_path)
    logger.info("Wrote %s", stats_path)
    return scored, stats


def main() -> None:
    try:
        scored, _stats = run()
    except Exception as exc:  # noqa: BLE001
        logger.error("%s", exc)
        sys.exit(1)
    print("=== Flavor Scout Step 5 OPPORTUNITY SCORING ===")
    print("Purchase intent used in score: false")
    print("Growth used in score:          false")
    print("Weights (renormalized 65):     demand 46.1538% | sentiment 30.7692% | brand fit 23.0769%")
    cols = [
        "rank",
        "flavor",
        "mentions",
        "demand_score",
        "sentiment_score",
        "brand_fit_score",
        "opportunity_score",
        "confidence",
    ]
    print(scored[cols].to_string(index=False))
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
