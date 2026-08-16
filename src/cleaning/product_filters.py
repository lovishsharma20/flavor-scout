"""Deterministic product/SKU filters shared by classification and the dashboard."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

_CONSUMABLE_SKU_RE = re.compile(
    r"\b("
    r"protein powder|whey|casein|isolate|creatine|collagen|bcaa|"
    r"electrolyte (drink|tablet|powder|mix|drops?)|"
    r"gummy|gummies|supplement|multivitamin|pre-?workout|"
    r"freeze[-\s]?dried|meal kit|tvp|textured vegetable|"
    r"drink mix|hydration tablets?"
    r")\b",
    re.I,
)
_GEAR_CONTAINER_SKU_RE = re.compile(
    r"\b("
    r"water bottle|sport bottle|sports bottle|gallon bottle|"
    r"shaker bottle|shaker cup|shaker|"
    r"tumbler|"
    r"hydration (pack|backpack|bladder|belt|reservoir|system|vest)|"
    r"hiking backpack|tactical (pack|bag|backpack)|daypack|backpack|"
    r"lunch (bag|box|pail|tote)|"
    r"cooler bag|soft cooler|cooler|"
    r"fanny pack|waist pack|sports bag|duffel|"
    r"foam roller|resistance band"
    r")\b",
    re.I,
)


def is_gear_container_sku(record: pd.Series | dict[str, Any]) -> bool:
    """True when the reviewed SKU is clearly gear/container, not a consumable."""
    blob = " ".join(
        [
            str(record.get("product_title") or ""),
            str(record.get("product_categories") or ""),
            str(record.get("main_category") or ""),
        ]
    )
    if _CONSUMABLE_SKU_RE.search(blob):
        return False
    return bool(_GEAR_CONTAINER_SKU_RE.search(blob))
