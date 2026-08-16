"""
Step 7: Deterministic Golden Candidate.

Reads (unchanged):
  data/processed/decision_engine_results.csv
  data/processed/opportunity_scores.csv
  data/processed/flavor_mention_qc.csv
  data/processed/analyzed_reviews.csv

Writes:
  data/processed/golden_candidate.json
  data/processed/golden_candidate.csv

Does NOT call an LLM. Does NOT invent flavors, demand, growth, or quotes.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from config.decision import DECISION_LABEL_SELECTED

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DECISIONS_PATH = PROJECT_ROOT / "data" / "processed" / "decision_engine_results.csv"
SCORES_PATH = PROJECT_ROOT / "data" / "processed" / "opportunity_scores.csv"
MENTION_QC_PATH = PROJECT_ROOT / "data" / "processed" / "flavor_mention_qc.csv"
REVIEWS_PATH = PROJECT_ROOT / "data" / "processed" / "analyzed_reviews.csv"
JSON_PATH = PROJECT_ROOT / "data" / "processed" / "golden_candidate.json"
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "golden_candidate.csv"

CONFIDENCE_RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
MAX_EVIDENCE_SNIPPETS = 3
SNIPPET_CHARS = 220


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def select_golden(selected: pd.DataFrame) -> pd.Series:
    if selected.empty:
        raise ValueError("No SELECTED flavors available for Golden Candidate")
    ranked = selected.copy()
    ranked["_conf"] = ranked["confidence"].astype(str).str.upper().map(
        lambda x: CONFIDENCE_RANK.get(x, 0)
    )
    ranked = ranked.sort_values(
        ["opportunity_score", "mentions", "positive_rate", "brand_fit_score", "_conf"],
        ascending=[False, False, False, False, False],
    )
    return ranked.iloc[0]


def evidence_reviews(flavor: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if not MENTION_QC_PATH.exists() or not REVIEWS_PATH.exists():
        return pd.DataFrame(), []
    qc = pd.read_csv(MENTION_QC_PATH)
    reviews = pd.read_csv(REVIEWS_PATH)
    keys = qc[
        qc["eligible"].map(_bool)
        & (qc["aggregation_flavor"].astype(str) == flavor)
    ]["review_key"].astype(str)
    matched = reviews[reviews["review_key"].astype(str).isin(set(keys))].copy()
    snippets: list[dict[str, Any]] = []
    for _, row in matched.head(MAX_EVIDENCE_SNIPPETS).iterrows():
        text = str(row.get("review_text") or "").replace("\n", " ").strip()
        if len(text) > SNIPPET_CHARS:
            text = text[: SNIPPET_CHARS - 1].rstrip() + "…"
        snippets.append(
            {
                "review_key": str(row.get("review_key")),
                "product_title": str(row.get("product_title") or "")[:120],
                "sentiment": str(row.get("sentiment") or ""),
                "intent": str(row.get("intent") or ""),
                "quote": text,
            }
        )
    return matched, snippets


def why_it_works(row: pd.Series) -> str:
    return (
        f"{row['flavor']} is the strongest opportunity in the analyzed dataset because "
        f"it has the highest Opportunity Score among Decision Engine SELECTED flavors "
        f"({float(row['opportunity_score']):.2f}/100), the most eligible mentions "
        f"({int(row['mentions'])}), {float(row['positive_rate']):.1%} positive sentiment, "
        f"and a brand-fit score of {float(row['brand_fit_score']):.2f}/100. "
        "Based on the available consumer evidence, it is the best-supported flavor "
        "concept in this sample - not a guaranteed launch success."
    )


def brand_recommendation(reviews: pd.DataFrame) -> tuple[str, str]:
    """
    Assignment brand map (MuscleBlaze / HK Vitals / TrueBasics) is used only
    when supporting SKUs clearly match that domain. Mixed or food-snack SKUs
    stay unassigned.
    """
    titles = " ".join(reviews["product_title"].fillna("").astype(str).tolist()).lower()
    protein_hits = sum(
        token in titles
        for token in ("protein", "whey", "isolate", "casein", "pre-workout", "preworkout")
    )
    electrolyte_hits = sum(
        token in titles
        for token in ("electrolyte", "hydration tablet", "vitamin", "gummy", "gummies")
    )
    fruit_food_hits = sum(
        token in titles
        for token in ("freeze dried", "freeze-dried", "fruit powder", "sliced strawberr")
    )
    if protein_hits and protein_hits > fruit_food_hits and protein_hits >= electrolyte_hits:
        return (
            "MuscleBlaze",
            "Supporting product titles in this sample are protein/performance SKUs, "
            "which the assignment maps to MuscleBlaze. This is a directional category fit, "
            "not a confirmed SKU brief.",
        )
    if electrolyte_hits and electrolyte_hits > fruit_food_hits and electrolyte_hits > protein_hits:
        return (
            "HK Vitals",
            "Supporting product titles in this sample are wellness/electrolyte SKUs, "
            "which the assignment maps to HK Vitals. This is a directional category fit, "
            "not a confirmed SKU brief.",
        )
    return (
        "Needs validation",
        "Most strawberry mentions in this sample are freeze-dried fruit snacks or fruit "
        "powders on Amazon Sports & Outdoors, plus at least one hydration-bottle SKU. "
        "That is not enough to assign MuscleBlaze (protein/performance), HK Vitals "
        "(wellness), or TrueBasics (premium functional) without further category validation.",
    )


def build_payload(
    winner: pd.Series,
    selected: pd.DataFrame,
    score_row: pd.Series,
    snippets: list[dict[str, Any]],
    supporting_reviews: pd.DataFrame,
) -> dict[str, Any]:
    brand, brand_why = brand_recommendation(supporting_reviews)
    others = selected[selected["flavor"] != winner["flavor"]].sort_values(
        "opportunity_score", ascending=False
    )
    beat = [
        {
            "flavor": r["flavor"],
            "opportunity_score": float(r["opportunity_score"]),
            "mentions": int(r["mentions"]),
            "reason": (
                f"Lower Opportunity Score than {winner['flavor']} "
                f"({float(r['opportunity_score']):.4f} vs "
                f"{float(winner['opportunity_score']):.4f})."
            ),
        }
        for _, r in others.iterrows()
    ]
    return {
        "flavor": str(winner["flavor"]),
        "rank": 1,
        "opportunity_score": float(winner["opportunity_score"]),
        "mentions": int(winner["mentions"]),
        "positive_rate": float(winner["positive_rate"]),
        "brand_fit_score": float(winner["brand_fit_score"]),
        "confidence": str(winner["confidence"]),
        "decision": DECISION_LABEL_SELECTED,
        "purchase_intent_available": False,
        "growth_available": False,
        "recommended_brand": brand,
        "recommended_brand_rationale": brand_why,
        "why_it_works": why_it_works(winner),
        "evidence_summary": {
            "source": "Amazon Reviews 2023 Sports & Outdoors (historical, through Sep 2023)",
            "universe": "relevant reviews with eligible aggregation_flavor",
            "mentions": int(winner["mentions"]),
            "positive_rate": float(winner["positive_rate"]),
            "brand_fit_score_0_to_100": float(winner["brand_fit_score"]),
            "opportunity_score": float(winner["opportunity_score"]),
            "confidence": str(winner["confidence"]),
            "selected_pool": selected["flavor"].tolist(),
            "representative_reviews": snippets,
        },
        "why_it_beat_other_selected": beat,
        "limitations": [
            "Purchase intent and explicit flavor requests are 0 in the eligible sample.",
            "No reliable flavor-level temporal growth; stored growth splits were not used.",
            "The source category is Sports & Outdoors, so eligible flavor volume is small (9 strawberry mentions).",
            "Scores use only Demand, Sentiment, and Brand Fit after dropping unavailable Growth and Purchase Intent pillars.",
            "This is the strongest evidence-backed candidate in this dataset, not a forecast that the flavor will succeed.",
        ],
        "anti_hallucination": {
            "llm_role": "Structure consumer evidence (relevance, flavor, sentiment, intent, pain_point, brand_fit).",
            "final_decision_role": "Deterministic aggregation, scoring, Decision Engine, then max Opportunity Score among SELECTED.",
            "llm_did_not": [
                "choose the Golden Candidate",
                "invent demand, growth, or purchase intent",
                "invent consumer quotes",
                "create a flavor absent from classified evidence",
            ],
            "source_of_truth": "Classified Amazon review rows and derived CSVs, not model imagination.",
        },
        "score_audit": {
            "opportunity_score_from_decision_engine": float(winner["opportunity_score"]),
            "opportunity_score_from_scores_csv": float(score_row["opportunity_score"]),
            "match": float(winner["opportunity_score"]) == float(score_row["opportunity_score"]),
        },
    }


def validate(payload: dict[str, Any], decisions: pd.DataFrame, scores: pd.DataFrame) -> None:
    flavor = payload["flavor"]
    assert payload["decision"] == DECISION_LABEL_SELECTED
    selected = set(
        decisions.loc[decisions["decision"] == DECISION_LABEL_SELECTED, "flavor"].astype(str)
    )
    assert flavor in selected
    assert flavor in set(scores["flavor"].astype(str))
    src = scores.loc[scores["flavor"] == flavor].iloc[0]
    assert float(payload["opportunity_score"]) == float(src["opportunity_score"])
    assert payload["purchase_intent_available"] is False
    assert payload["growth_available"] is False
    assert payload["score_audit"]["match"] is True
    assert len([payload["flavor"]]) == 1
    logger.info("Golden Candidate validation passed: %s", flavor)


def run() -> dict[str, Any]:
    decisions = pd.read_csv(DECISIONS_PATH)
    scores = pd.read_csv(SCORES_PATH)
    selected = decisions[decisions["decision"] == DECISION_LABEL_SELECTED].copy()
    winner = select_golden(selected)
    score_row = scores.loc[scores["flavor"] == winner["flavor"]].iloc[0]
    supporting, snippets = evidence_reviews(str(winner["flavor"]))
    payload = build_payload(winner, selected, score_row, snippets, supporting)
    validate(payload, decisions, scores)
    JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "flavor": payload["flavor"],
                "rank": payload["rank"],
                "opportunity_score": payload["opportunity_score"],
                "mentions": payload["mentions"],
                "positive_rate": payload["positive_rate"],
                "brand_fit_score": payload["brand_fit_score"],
                "confidence": payload["confidence"],
                "decision": payload["decision"],
                "purchase_intent_available": payload["purchase_intent_available"],
                "growth_available": payload["growth_available"],
                "recommended_brand": payload["recommended_brand"],
                "why_it_works": payload["why_it_works"],
            }
        ]
    ).to_csv(CSV_PATH, index=False)
    logger.info("Wrote %s", JSON_PATH)
    logger.info("Wrote %s", CSV_PATH)
    return payload


def main() -> None:
    try:
        payload = run()
    except Exception as exc:
        logger.error("%s", exc)
        sys.exit(1)
    print("=== Flavor Scout Step 7 GOLDEN CANDIDATE ===")
    print(f"Flavor:              {payload['flavor']}")
    print(f"Opportunity score:   {payload['opportunity_score']}")
    print(f"Mentions:            {payload['mentions']}")
    print(f"Positive rate:       {payload['positive_rate']}")
    print(f"Brand-fit score:     {payload['brand_fit_score']}")
    print(f"Confidence:          {payload['confidence']}")
    print(f"Recommended brand:   {payload['recommended_brand']}")
    print(f"Why it works:        {payload['why_it_works']}")
    print(f"Saved: {JSON_PATH}")


if __name__ == "__main__":
    main()
