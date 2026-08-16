"""
Step 6: Deterministic Decision Engine (SELECTED vs REJECTED).

Reads:  data/processed/opportunity_scores.csv  (unchanged)
Writes: data/processed/decision_engine_results.csv
        data/processed/decision_engine_stats.json

Does NOT call an LLM. Does NOT choose a Golden Candidate.
Does NOT modify opportunity scores.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from config.decision import (
    DECISION_BRAND_FIT_SCORE_MIN,
    DECISION_CONFIDENCE_ALLOWED,
    DECISION_LABEL_REJECTED,
    DECISION_LABEL_SELECTED,
    DECISION_MENTIONS_MIN,
    DECISION_OPPORTUNITY_SCORE_MIN,
    DECISION_POSITIVE_RATE_MIN,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCORES_PATH = PROJECT_ROOT / "data" / "processed" / "opportunity_scores.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "decision_engine_results.csv"
STATS_PATH = PROJECT_ROOT / "data" / "processed" / "decision_engine_stats.json"

LIMITATION_NOTE = (
    "Purchase intent and temporal growth were unavailable in this dataset; "
    "those gaps were not used as automatic reject reasons."
)


def thresholds() -> dict[str, Any]:
    return {
        "opportunity_score_min": DECISION_OPPORTUNITY_SCORE_MIN,
        "mentions_min": DECISION_MENTIONS_MIN,
        "positive_rate_min": DECISION_POSITIVE_RATE_MIN,
        "brand_fit_score_min": DECISION_BRAND_FIT_SCORE_MIN,
        "confidence_allowed": list(DECISION_CONFIDENCE_ALLOWED),
        "note": (
            "Project decision thresholds, not values supplied by Amazon Reviews 2023. "
            "brand_fit_score_min is 50 on the 0–100 scale stored in opportunity_scores.csv "
            "(same rule as 0.50 on the 0–1 aggregation scale)."
        ),
    }


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def evaluate_row(rec: pd.Series) -> tuple[str, str, list[str], list[str]]:
    score = float(rec["opportunity_score"])
    mentions = int(rec["mentions"])
    positive_rate = float(rec["positive_rate"])
    brand_fit = float(rec["brand_fit_score"])
    confidence = str(rec["confidence"]).strip().upper()

    passed: list[str] = []
    failed: list[str] = []

    if score >= DECISION_OPPORTUNITY_SCORE_MIN:
        passed.append(
            f"opportunity score {score:.4f} >= {DECISION_OPPORTUNITY_SCORE_MIN}"
        )
    else:
        failed.append(
            f"opportunity score {score:.4f} < {DECISION_OPPORTUNITY_SCORE_MIN}"
        )

    if mentions >= DECISION_MENTIONS_MIN:
        passed.append(f"mentions {mentions} >= {DECISION_MENTIONS_MIN}")
    else:
        failed.append(f"mentions {mentions} < {DECISION_MENTIONS_MIN}")

    if positive_rate >= DECISION_POSITIVE_RATE_MIN:
        passed.append(
            f"positive rate {positive_rate:.4f} >= {DECISION_POSITIVE_RATE_MIN}"
        )
    else:
        failed.append(
            f"positive rate {positive_rate:.4f} < {DECISION_POSITIVE_RATE_MIN}"
        )

    if brand_fit >= DECISION_BRAND_FIT_SCORE_MIN:
        passed.append(
            f"brand-fit {brand_fit:.4f} >= {DECISION_BRAND_FIT_SCORE_MIN}"
        )
    else:
        failed.append(
            f"brand-fit {brand_fit:.4f} < {DECISION_BRAND_FIT_SCORE_MIN}"
        )

    if confidence in DECISION_CONFIDENCE_ALLOWED:
        passed.append(f"confidence {confidence} is allowed")
    else:
        failed.append(
            f"confidence {confidence} is not in {list(DECISION_CONFIDENCE_ALLOWED)}"
        )

    if failed:
        decision = DECISION_LABEL_REJECTED
        reason = (
            "REJECTED because "
            + "; ".join(failed)
            + ". "
            + LIMITATION_NOTE
        )
    else:
        decision = DECISION_LABEL_SELECTED
        reason = (
            "SELECTED because opportunity score is above threshold, "
            "the flavor has sufficient mentions, positive sentiment, "
            "acceptable brand fit, and HIGH or MEDIUM confidence. "
            + LIMITATION_NOTE
        )
    return decision, reason, passed, failed


def decide(scores: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    eligible = scores[scores["eligible_for_scoring"].map(_bool)].copy()
    if len(eligible) != 11:
        raise ValueError(f"Expected 11 scoring-eligible flavors, found {len(eligible)}")
    if eligible["flavor"].duplicated().any():
        raise ValueError("Duplicate flavors in scoring input")

    rows: list[dict[str, Any]] = []
    for _, rec in eligible.iterrows():
        decision, reason, _passed, failed = evaluate_row(rec)
        rows.append(
            {
                "flavor": rec["flavor"],
                "opportunity_score": rec["opportunity_score"],
                "confidence": rec["confidence"],
                "mentions": int(rec["mentions"]),
                "positive_rate": rec["positive_rate"],
                "brand_fit_score": rec["brand_fit_score"],
                "decision": decision,
                "decision_reason": reason,
                "purchase_intent_available": False,
                "growth_available": False,
                "thresholds_used": json.dumps(thresholds(), sort_keys=True),
                "fail_count": len(failed),
            }
        )

    result = pd.DataFrame(rows)
    result["_selected"] = result["decision"] == DECISION_LABEL_SELECTED
    result = result.sort_values(
        ["_selected", "opportunity_score"],
        ascending=[False, False],
    ).reset_index(drop=True)
    result = result.drop(columns=["_selected", "fail_count"])

    selected = result[result["decision"] == DECISION_LABEL_SELECTED]
    rejected = result[result["decision"] == DECISION_LABEL_REJECTED]
    stats = {
        "total_candidates": int(len(result)),
        "selected_count": int(len(selected)),
        "rejected_count": int(len(rejected)),
        "thresholds": thresholds(),
        "unavailable_evidence": {
            "purchase_intent": False,
            "growth": False,
            "note": (
                "purchase_intent_count and request_count are 0; flavor-level "
                "growth is not reliable. Missing evidence did not auto-reject."
            ),
        },
        "top_selected_candidates": selected["flavor"].tolist(),
        "rejected_candidates": rejected["flavor"].tolist(),
    }
    return result, stats


def validate(result: pd.DataFrame, scores: pd.DataFrame) -> None:
    assert len(result) == 11
    assert result["flavor"].nunique() == 11
    assert set(result["flavor"]) == set(
        scores.loc[scores["eligible_for_scoring"].map(_bool), "flavor"]
    )
    assert result["decision"].isin(
        [DECISION_LABEL_SELECTED, DECISION_LABEL_REJECTED]
    ).all()
    assert result.groupby("flavor").size().max() == 1
    assert not result["decision"].isna().any()
    assert not result["purchase_intent_available"].any()
    assert not result["growth_available"].any()
    merged = result.merge(
        scores[["flavor", "opportunity_score"]],
        on="flavor",
        suffixes=("_out", "_src"),
    )
    assert (merged["opportunity_score_out"] == merged["opportunity_score_src"]).all()
    logger.info("Decision Engine validation passed.")


def run(
    scores_path: Path = SCORES_PATH,
    output_path: Path = OUTPUT_PATH,
    stats_path: Path = STATS_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    scores = pd.read_csv(scores_path)
    result, stats = decide(scores)
    validate(result, scores)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("Wrote %s", output_path)
    logger.info("Wrote %s", stats_path)
    return result, stats


def main() -> None:
    try:
        result, stats = run()
    except Exception as exc:
        logger.error("%s", exc)
        sys.exit(1)
    print("=== Flavor Scout Step 6 DECISION ENGINE ===")
    print(f"Total candidates: {stats['total_candidates']}")
    print(f"SELECTED:         {stats['selected_count']}")
    print(f"REJECTED:         {stats['rejected_count']}")
    print(f"Thresholds:       {json.dumps(stats['thresholds'], indent=2)}")
    cols = [
        "flavor",
        "opportunity_score",
        "confidence",
        "mentions",
        "positive_rate",
        "brand_fit_score",
        "decision",
    ]
    print(result[cols].to_string(index=False))
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
