"""Pydantic schemas for Flavor Scout LLM classification output."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

Sentiment = Literal["positive", "neutral", "negative"]
ConsumerIntent = Literal[
    "request",
    "preference",
    "complaint",
    "praise",
    "comparison",
    "general_mention",
    "purchase_intent",
]
BrandFit = Literal["strong", "moderate", "weak", "none"]
Confidence = Literal["high", "medium", "low"]

_INTENT_ALIASES = {
    "request": "request",
    "preference": "preference",
    "complaint": "complaint",
    "praise": "praise",
    "recommendation": "praise",
    "comparison": "comparison",
    "general_mention": "general_mention",
    "other": "general_mention",
    "none": "general_mention",
    "purchase_intent": "purchase_intent",
    "purchase": "purchase_intent",
}

_BRAND_FIT_ALLOWED = {"strong", "moderate", "weak", "none"}
_BRAND_FIT_ALIASES = {
    "strong": "strong",
    "moderate": "moderate",
    "weak": "weak",
    "none": "none",
    "no": "none",
    "n/a": "none",
    "na": "none",
    "not_applicable": "none",
    "unrelated": "none",
}
_BRAND_NAME_LABELS = {
    "muscleblaze",
    "muscle blaze",
    "hk vitals",
    "hkvitals",
    "truebasics",
    "true basics",
    "healthkart",
    "health kart",
}

class ReviewAnalysis(BaseModel):
    """Structured LLM classification for one Amazon review (Step 3)."""

    relevant: bool = Field(
        ...,
        description=(
            "True if the review is relevant to food, beverages, supplements, protein, "
            "electrolytes, wellness products, or flavor discovery. False for unrelated "
            "gear such as hydration backpacks, tote bags, apparel, or equipment."
        ),
    )
    flavor: Optional[str] = Field(
        default=None,
        description="Specific flavor named, requested, preferred, or criticized. Null if none.",
    )
    sentiment: Sentiment = Field(
        ...,
        description="Overall sentiment of the review.",
    )
    intent: ConsumerIntent = Field(
        ...,
        description="Primary consumer intent.",
    )
    pain_point: Optional[str] = Field(
        default=None,
        description="Actual consumer problem if stated. Null if none; do not invent.",
    )
    brand_fit: BrandFit = Field(
        ...,
        description=(
            "Fit strength to HealthKart fitness-nutrition context "
            "(MuscleBlaze / HK Vitals / TrueBasics style products). "
            "Must be strong, moderate, weak, or none — never a brand name. "
            "Use none if the product is unrelated. Do not invent a HealthKart brand or SKU."
        ),
    )
    reasoning: str = Field(
        ...,
        description="Short justification for the classification, not a review summary.",
    )
    confidence: Confidence = Field(
        ...,
        description="Classifier confidence.",
    )

    @field_validator("flavor", "pain_point", mode="before")
    @classmethod
    def empty_str_to_none(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("intent", mode="before")
    @classmethod
    def normalize_intent(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        key = value.strip().lower().replace(" ", "_").replace("-", "_")
        return _INTENT_ALIASES.get(key, key)

    @field_validator("brand_fit", mode="before")
    @classmethod
    def normalize_brand_fit(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        raw = " ".join(value.strip().lower().replace("_", " ").replace("-", " ").split())
        compact = raw.replace(" ", "")
        if raw in _BRAND_FIT_ALLOWED:
            return raw
        if raw in _BRAND_FIT_ALIASES:
            return _BRAND_FIT_ALIASES[raw]

        for score in ("strong", "moderate", "weak", "none"):
            if raw == score or raw.startswith(score + " ") or raw.endswith(" " + score):
                return score
        if raw in _BRAND_NAME_LABELS or compact in {item.replace(" ", "") for item in _BRAND_NAME_LABELS}:
            raise ValueError(
                "brand_fit must be one of strong|moderate|weak|none, not a brand name; "
                "put MuscleBlaze / HK Vitals / TrueBasics in reasoning only"
            )
        return raw

    @field_validator("sentiment", "confidence", mode="before")
    @classmethod
    def lowercase_enums(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("reasoning", mode="before")
    @classmethod
    def strip_reasoning(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value
