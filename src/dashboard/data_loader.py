"""Load completed Flavor Scout analysis artifacts. No LLM/API calls."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"

REQUIRED = {
    "trends": PROCESSED / "flavor_trends.csv",
    "trend_stats": PROCESSED / "flavor_trends_stats.json",
    "scores": PROCESSED / "opportunity_scores.csv",
    "decisions": PROCESSED / "decision_engine_results.csv",
    "golden": PROCESSED / "golden_candidate.json",
}


# Presentation-only extra filter for container SKUs that slipped past older
# classifications. Does not rewrite analyzed_reviews.csv.
_EVIDENCE_GEAR_RE = re.compile(
    r"\b(flask|softflask|hydrapak|soft flask|gel flask|water bottle|sport bottle|"
    r"shaker|tumbler|hydration (pack|backpack|bladder))\b",
    re.I,
)


class DashboardDataError(Exception):
    """User-facing load failure."""


@dataclass
class DashboardData:
    trends: pd.DataFrame
    scores: pd.DataFrame
    decisions: pd.DataFrame
    board: pd.DataFrame
    golden: dict[str, Any]
    stats: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DashboardDataError(
            f"Could not read {path.name}. Re-run the analysis pipeline or restore the file."
        ) from exc
    if not isinstance(payload, dict):
        raise DashboardDataError(f"{path.name} is not a valid analysis summary.")
    return payload


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DashboardDataError(
            f"Could not read {path.name}. The dashboard needs completed analysis files."
        ) from exc
    if df.empty:
        raise DashboardDataError(f"{path.name} is empty.")
    return df


def load_dashboard_data() -> DashboardData:
    missing = [name for name, path in REQUIRED.items() if not path.exists()]
    if missing:
        labels = ", ".join(REQUIRED[name].name for name in missing)
        raise DashboardDataError(
            "Some completed analysis files are missing: "
            f"{labels}. Run trend aggregation, scoring, the Decision Engine, "
            "and Golden Candidate before opening the dashboard."
        )
    trends = _read_csv(REQUIRED["trends"])
    scores = _read_csv(REQUIRED["scores"])
    decisions = _read_csv(REQUIRED["decisions"])
    stats = _read_json(REQUIRED["trend_stats"])
    golden = _read_json(REQUIRED["golden"])
    for frame, cols in (
        (trends, ("flavor", "mentions", "positive_rate")),
        (scores, ("flavor", "opportunity_score", "confidence", "mentions", "positive_rate", "brand_fit_score")),
        (decisions, ("flavor", "decision", "opportunity_score", "confidence")),
    ):
        absent = [c for c in cols if c not in frame.columns]
        if absent:
            raise DashboardDataError(
                "Analysis files are missing expected columns: " + ", ".join(absent)
            )
    board = scores.merge(
        decisions[["flavor", "decision", "decision_reason"]],
        on="flavor",
        how="left",
    )
    if "decision" not in board.columns or board["decision"].isna().any():
        raise DashboardDataError("Decision Engine results do not cover every scored flavor.")
    return DashboardData(
        trends=trends,
        scores=scores,
        decisions=decisions,
        board=board,
        golden=golden,
        stats=stats,
    )


def load_review_evidence(flavor: str, limit: int = 5) -> pd.DataFrame:
    """Load a few classified reviews for one eligible flavor. Cached by the UI."""
    qc_path = PROCESSED / "flavor_mention_qc.csv"
    reviews_path = PROCESSED / "analyzed_reviews.csv"
    if not qc_path.exists() or not reviews_path.exists():
        return pd.DataFrame()
    qc = pd.read_csv(
        qc_path,
        usecols=["review_key", "aggregation_flavor", "eligible"],
    )
    keep = qc["eligible"].astype(str).str.lower().isin(["true", "1"]) & (
        qc["aggregation_flavor"].astype(str) == flavor
    )
    keys = set(qc.loc[keep, "review_key"].astype(str))
    if not keys:
        return pd.DataFrame()
    wanted = [
        "review_key",
        "product_title",
        "product_categories",
        "main_category",
        "review_text",
        "relevant",
        "flavor",
        "sentiment",
        "intent",
        "pain_point",
        "brand_fit",
        "confidence",
    ]
    header = pd.read_csv(reviews_path, nrows=0)
    cols = [c for c in wanted if c in header.columns]
    reviews = pd.read_csv(reviews_path, usecols=cols)
    matched = reviews[reviews["review_key"].astype(str).isin(keys)].copy()
    if "relevant" in matched.columns:
        matched = matched[
            matched["relevant"].astype(str).str.lower().isin(["true", "1", "yes"])
        ]
    from src.classification.classify_reviews import is_gear_container_sku

    if not matched.empty:
        gear = matched.apply(is_gear_container_sku, axis=1)
        title_gear = matched["product_title"].fillna("").astype(str).map(
            lambda t: bool(_EVIDENCE_GEAR_RE.search(t))
        )
        matched = matched.loc[~gear & ~title_gear]
    return matched.head(limit)
