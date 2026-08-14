"""Pydantic schemas for Flavor Scout LLM classification output."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


Category = Literal[
    "preference",
    "complaint",
    "request",
    "recommendation",
    "comparison",
    "other",
]
Sentiment = Literal["positive", "negative", "neutral", "mixed"]
Intent = Literal[
    "share_preference",
    "complain",
    "request_flavor",
    "recommend",
    "compare",
    "ask_question",
    "other",
]
PurchaseIntent = Literal["none", "low", "medium", "high"]
BrandFit = Literal["strong", "moderate", "weak", "unknown"]


class CommentAnalysis(BaseModel):
    """Structured LLM classification for one Reddit comment/post."""

    relevant: bool = Field(
        ...,
        description="True if the text discusses flavors, taste, or related product feedback.",
    )
    flavor: Optional[str] = Field(
        default=None,
        description="Primary flavor mentioned (e.g. chocolate, vanilla). Null if none.",
    )
    category: Category = Field(
        ...,
        description="High-level discussion type.",
    )
    sentiment: Sentiment = Field(
        ...,
        description="Overall sentiment toward taste/flavor/product.",
    )
    intent: Intent = Field(
        ...,
        description="What the author is trying to do.",
    )
    purchase_intent: PurchaseIntent = Field(
        ...,
        description="Signal that the author may buy or switch products.",
    )
    pain_point: Optional[str] = Field(
        default=None,
        description="Short pain point if present (e.g. too sweet, chalky). Null if none.",
    )
    brand_fit: BrandFit = Field(
        ...,
        description="How well this insight could inform a flavor/product brand decision.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence between 0 and 1.",
    )

    @field_validator("flavor", "pain_point", mode="before")
    @classmethod
    def empty_str_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, value: object) -> object:
        if isinstance(value, (int, float)):
            return max(0.0, min(1.0, float(value)))
        return value
