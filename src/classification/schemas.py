"""Pydantic schemas for Flavor Scout LLM classification output."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


Sentiment = Literal["positive", "neutral", "negative"]
ConsumerIntent = Literal[
    "preference",
    "request",
    "complaint",
    "purchase_intent",
    "recommendation",
    "other",
    "none",
]
BrandFit = Literal["strong", "moderate", "weak", "none"]
Confidence = Literal["high", "medium", "low"]


class ReviewAnalysis(BaseModel):
    """Structured LLM classification for one Amazon review (Step 3)."""

    relevant: bool = Field(
        ...,
        description=(
            "True only if the review is useful for Flavor Scout flavor discovery "
            "in sports nutrition (protein, whey, supplements, electrolytes, "
            "gummies, pre-workout, etc.). False for unrelated gear such as "
            "hydration backpacks, tote bags, apparel, or equipment."
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
            "Fit to HealthKart fitness-nutrition context (MuscleBlaze / HK Vitals / "
            "TrueBasics style products). none if the product is unrelated. "
            "Do not invent a HealthKart brand name."
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

    @field_validator("reasoning", mode="before")
    @classmethod
    def strip_reasoning(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value
