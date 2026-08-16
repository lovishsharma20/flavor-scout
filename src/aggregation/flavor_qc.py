"""
Deterministic flavor eligibility and normalization for Trend Aggregation.

Does not modify LLM `flavor`. Produces `aggregation_flavor` (or None).
"""

from __future__ import annotations

import re
from typing import Any

# Exact phrases after lowercase/whitespace normalization.
EXCLUDED_MEAL_DISH = {
    "beef stroganoff",
    "sweet & sour pork",
    "sweet and sour pork",
    "rice and chicken",
    "pad thai",
    "cheesy hamburger",
    "cheese enchilada ranchero",
    "red and green bell pepper",
    "bell pepper",
    "taco",
    "chicken",
    "shrimp",
    "beefish",
    "chickenish",
    "beef",
    "pork",
    "cheese",
    "colby cheese",
    "cheddar cheese",
    "oatmeal quinoa apple cinnamon",
    "turkey dinner",
    "sweet pork",
    "homestyle chicken",
    "garlic",
    "scrambled eggs",
    "mashed potatoes",
}

EXCLUDED_GENERIC = {
    "unflavored",
    "original",
    "sour",
    "fruit",
    "whey milk",
    "subtle citrus",
    "none",
    "n/a",
    "na",
    "unknown",
}

_MEAL_MARKERS = (
    "stroganoff",
    "pad thai",
    "enchilada",
    "hamburger",
    "dinner",
    "homestyle",
    "entrée",
    "entree",
    "casserole",
    "mashed",
    "scrambled",
    "rice and",
    "sweet & sour",
    "sweet and sour",
)

# Obvious plural → singular only. Do not merge distinct flavor families.
_PLURAL_MAP = {
    "strawberries": "strawberry",
    "blueberries": "blueberry",
    "bananas": "banana",
    "oranges": "orange",
    "coconuts": "coconut",
    "lemons": "lemon",
    "peaches": "peach",
    "apples": "apple",
    "raspberries": "raspberry",
    "mangoes": "mango",
    "mangos": "mango",
    "cranberries": "cranberry",
}

_WS_RE = re.compile(r"\s+")


def normalize_flavor_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("_", " ")
    if not text or text in {"nan", "none", "null"}:
        return None
    text = text.replace("&", " and ")
    text = text.replace("-", " ")
    text = _WS_RE.sub(" ", text).strip()
    return text or None


def apply_variant_normalization(normalized: str) -> tuple[str, str | None]:
    """Return (canonical, merge_rule_or_none)."""
    if normalized in _PLURAL_MAP:
        return _PLURAL_MAP[normalized], f"{normalized}->{_PLURAL_MAP[normalized]}"
    return normalized, None


def exclusion_reason(canonical: str) -> str | None:
    if canonical in EXCLUDED_GENERIC:
        return "generic_non_launchable"
    if canonical in EXCLUDED_MEAL_DISH:
        return "meal_or_dish"
    if any(marker in canonical for marker in _MEAL_MARKERS):
        return "meal_or_dish"
    return None


def aggregation_flavor(raw_flavor: object) -> tuple[str | None, str | None, str | None]:
    """
    Returns (aggregation_flavor, exclusion_reason, merge_rule).
    aggregation_flavor is title-cased for display when eligible.
    """
    normalized = normalize_flavor_text(raw_flavor)
    if not normalized:
        return None, "empty", None
    canonical, merge_rule = apply_variant_normalization(normalized)
    reason = exclusion_reason(canonical)
    if reason:
        return None, reason, merge_rule
    display = " ".join(part.capitalize() for part in canonical.split())
    return display, None, merge_rule


def annotate_row(row: dict[str, Any] | Any) -> dict[str, Any]:
    raw = row.get("flavor") if hasattr(row, "get") else None
    flavor, reason, merge = aggregation_flavor(raw)
    return {
        "aggregation_flavor": flavor,
        "exclusion_reason": reason,
        "normalization_merge": merge or "",
    }
