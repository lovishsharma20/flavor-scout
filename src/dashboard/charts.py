"""Trend Wall chart. No fabricated growth series."""

from __future__ import annotations

import altair as alt
import pandas as pd


SELECTED_COLOR = "#1B4D3E"
REJECTED_COLOR = "#9A8F84"
FOCUS_COLOR = "#C45C26"


def mentions_bar_chart(board: pd.DataFrame, focus_flavor: str) -> alt.Chart:
    plot = board.copy()
    plot["Mentions"] = plot["mentions"].astype(int)
    plot["Status"] = plot["decision"].astype(str).str.title()
    plot["Focus"] = plot["flavor"].astype(str) == str(focus_flavor)
    plot = plot.sort_values("Mentions", ascending=True)
    return (
        alt.Chart(plot)
        .mark_bar(size=18, cornerRadiusEnd=3)
        .encode(
            x=alt.X("Mentions:Q", title="Eligible mentions"),
            y=alt.Y("flavor:N", sort="-x", title=None),
            color=alt.Color(
                "Status:N",
                scale=alt.Scale(
                    domain=["Selected", "Rejected"],
                    range=[SELECTED_COLOR, REJECTED_COLOR],
                ),
                legend=alt.Legend(title="Decision", orient="bottom"),
            ),
            opacity=alt.condition("datum.Focus", alt.value(1.0), alt.value(0.55)),
            tooltip=[
                alt.Tooltip("flavor:N", title="Flavor"),
                alt.Tooltip("Mentions:Q", title="Mentions"),
                alt.Tooltip("opportunity_score:Q", title="Opportunity score", format=".2f"),
                alt.Tooltip("positive_rate:Q", title="Positive sentiment", format=".1%"),
                alt.Tooltip("Status:N", title="Decision"),
                alt.Tooltip("confidence:N", title="Confidence"),
            ],
        )
        .properties(height=max(280, 28 * len(plot)), title="Scoring candidates by mention volume")
        .configure_axis(labelFontSize=13, titleFontSize=12, grid=False)
        .configure_view(strokeWidth=0)
        .configure_title(fontSize=15, fontWeight=600, anchor="start")
    )
