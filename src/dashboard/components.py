"""Reusable Streamlit UI blocks."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


def _label(value: object, empty: str = "None recorded") -> str:
    if value is None:
        return empty
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "nat"}:
        return empty
    return text


def kpi_row(stats: dict[str, Any], selected_count: int) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviews analyzed", f"{int(stats.get('classified_reviews') or 7931):,}")
    c2.metric("Relevant reviews", f"{int(stats.get('relevant_reviews') or 0):,}")
    c3.metric("Eligible flavor mentions", f"{int(stats.get('eligible_flavor_mentions') or 0):,}")
    c4.metric("Selected opportunities", str(selected_count))


def flavor_metric_strip(row: pd.Series) -> None:
    a, b, c, d, e = st.columns(5)
    a.metric("Mentions", int(row["mentions"]))
    b.metric("Positive sentiment", f"{float(row['positive_rate']) * 100:.1f}%")
    c.metric("Opportunity score", f"{float(row['opportunity_score']):.2f}")
    d.metric("Brand-fit score", f"{float(row['brand_fit_score']):.2f}")
    e.metric("Confidence", str(row["confidence"]))


def evidence_cards(reviews: pd.DataFrame, fallback: list[dict[str, Any]] | None = None) -> None:
    if reviews is not None and not reviews.empty:
        for _, rec in reviews.iterrows():
            title = str(rec.get("product_title") or "Reviewed product")
            with st.container(border=True):
                st.markdown(f"**{title}**")
                text = str(rec.get("review_text") or "").replace("\n", " ").strip()
                if len(text) > 360:
                    text = text[:359].rstrip() + "…"
                st.write(text)
                m1, m2, m3, m4 = st.columns(4)
                m1.caption(f"Relevance: {_label(rec.get('relevant'))}")
                m2.caption(f"Flavor: {_label(rec.get('flavor'))}")
                m3.caption(f"Sentiment: {_label(rec.get('sentiment'))}")
                m4.caption(f"Intent: {_label(rec.get('intent'))}")
                st.caption(
                    f"Pain point: {_label(rec.get('pain_point'))} · "
                    f"Brand fit: {_label(rec.get('brand_fit'))} · "
                    f"Confidence: {_label(rec.get('confidence'))}"
                )
        return
    if fallback:
        for item in fallback:
            with st.container(border=True):
                st.markdown(f"**{item.get('product_title') or 'Reviewed product'}**")
                st.write(item.get("quote") or "Review text unavailable.")
                st.caption(
                    f"Sentiment: {item.get('sentiment')} · Intent: {item.get('intent')}"
                )
        return
    st.info("No stored review excerpts are available for this flavor.")
