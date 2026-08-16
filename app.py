"""
Flavor Scout Streamlit dashboard.

Presentation layer only. Analysis is already complete offline.
Does not call an LLM or any external API.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dashboard.charts import mentions_bar_chart
from src.dashboard.components import evidence_cards, flavor_metric_strip, kpi_row
from src.dashboard.data_loader import DashboardDataError, load_dashboard_data, load_review_evidence
from src.dashboard.methodology import HOW_IT_WORKS, LAYOUT_RATIONALE, PIPELINE_STEPS

st.set_page_config(
    page_title="Flavor Scout | HealthKart",
    page_icon="FS",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
    .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px; }
    h1, h2, h3 { letter-spacing: -0.02em; }
    .fs-kicker { font-size: 0.78rem; font-weight: 600; letter-spacing: 0.12em;
        text-transform: uppercase; color: #6B645C; margin-bottom: 0.2rem; }
    .fs-subtitle { color: #4A453F; font-size: 1.05rem; margin-bottom: 0.35rem; }
    .fs-meta { color: #6B645C; font-size: 0.92rem; }
    .gc-wrap { background: #13261F; color: #F4EFE8; padding: 1.6rem 1.8rem 1.4rem;
        border-radius: 16px; margin: 0.4rem 0 1rem; }
    .gc-wrap h2 { color: #F4EFE8 !important; font-size: 2.15rem; margin: 0.15rem 0 0.4rem; }
    .gc-score { font-size: 1.35rem; font-weight: 650; color: #E7C9A4; }
    .gc-line { color: #D9D0C6; font-size: 1rem; margin-top: 0.45rem; }
    .status-pill { display: inline-block; background: #2F6F4E; color: #F4EFE8;
        font-size: 0.75rem; font-weight: 600; letter-spacing: 0.06em;
        padding: 0.2rem 0.55rem; border-radius: 999px; }
    .warn-note { background: #F3E6D4; color: #5C3B16; padding: 0.7rem 0.9rem;
        border-radius: 10px; font-size: 0.92rem; }
    .pipe { display: flex; flex-wrap: wrap; gap: 0.4rem; align-items: center;
        font-size: 0.88rem; color: #3F3A36; }
    .pipe span { background: #EFEAE3; padding: 0.35rem 0.55rem; border-radius: 8px; }
    .pipe em { font-style: normal; color: #8A8178; }
    div[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #E6E0D8;
        border-radius: 12px; padding: 0.7rem 0.85rem; }
</style>
"""


@st.cache_data(show_spinner=False)
def _cached_bundle():
    return load_dashboard_data()


@st.cache_data(show_spinner=False)
def _cached_evidence(flavor: str) -> pd.DataFrame:
    return load_review_evidence(flavor, limit=5)


def _init_state(default_flavor: str) -> None:
    if "selected_flavor" not in st.session_state:
        st.session_state.selected_flavor = default_flavor


def _select(flavor: str) -> None:
    st.session_state.selected_flavor = flavor


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    try:
        data = _cached_bundle()
    except DashboardDataError as exc:
        st.markdown('<div class="fs-kicker">Flavor Scout</div>', unsafe_allow_html=True)
        st.title("Dashboard unavailable")
        st.error(str(exc))
        st.caption("No live model calls are made from this app. Analysis must already exist on disk.")
        return
    except Exception:
        st.title("Dashboard unavailable")
        st.error("The analysis files could not be loaded. Please restore the completed outputs and refresh.")
        return

    golden = data.golden
    default_flavor = str(golden.get("flavor") or "Strawberry")
    _init_state(default_flavor)
    board = data.board.copy()
    flavors = board["flavor"].astype(str).tolist()
    if st.session_state.selected_flavor not in flavors:
        st.session_state.selected_flavor = default_flavor
    focus = st.session_state.selected_flavor
    selected = board[board["decision"] == "SELECTED"].sort_values(
        "opportunity_score", ascending=False
    )
    rejected = board[board["decision"] == "REJECTED"].sort_values(
        "opportunity_score", ascending=False
    )

    left, right = st.columns([3.2, 1.1])
    with left:
        st.markdown('<div class="fs-kicker">Flavor Scout</div>', unsafe_allow_html=True)
        st.title("Consumer intelligence for HealthKart's next flavor")
        st.markdown(
            '<p class="fs-subtitle">Amazon Reviews • AI-assisted analysis • Deterministic scoring</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="fs-meta"><span class="status-pill">Analysis complete</span>'
            "&nbsp; Dataset analyzed — not a live feed.</p>",
            unsafe_allow_html=True,
        )
    with right:
        with st.expander("How it works", expanded=False):
            st.write(HOW_IT_WORKS)
            st.caption(LAYOUT_RATIONALE)

    st.markdown("### Market pulse")
    kpi_row(data.stats, int(len(selected)))
    st.caption(
        f"Scoring candidates: {int(data.stats.get('flavors_eligible_for_scoring') or len(board))} · "
        "Purchase/request signals: none detected · Growth signal unavailable"
    )

    st.markdown("### Trend Wall")
    st.caption(
        "What consumers in this dataset mentioned as eligible flavor concepts. "
        "A ranked bar chart is used so magnitudes can be compared directly."
    )
    st.altair_chart(mentions_bar_chart(board, focus), width="stretch")
    st.markdown(
        '<div class="warn-note">Growth signal unavailable: insufficient reliable temporal evidence. '
        "No explicit purchase/request signals detected.</div>",
        unsafe_allow_html=True,
    )
    st.selectbox("Inspect a flavor", options=flavors, key="selected_flavor")
    focus = st.session_state.selected_flavor
    focus_row = board.loc[board["flavor"].astype(str) == focus].iloc[0]
    flavor_metric_strip(focus_row)
    st.caption(str(focus_row.get("decision_reason") or ""))

    st.markdown("### Decision Engine")
    st.caption("Viable opportunities versus ideas that did not clear the project thresholds.")
    col_s, col_r = st.columns(2)
    with col_s:
        st.markdown("**Selected**")
        for _, row in selected.iterrows():
            label = f"{row['flavor']}  ·  {float(row['opportunity_score']):.2f}  ·  {row['confidence']}"
            st.button(
                label,
                key=f"sel_{row['flavor']}",
                use_container_width=True,
                type="primary" if row["flavor"] == focus else "secondary",
                on_click=_select,
                args=(str(row["flavor"]),),
            )
    with col_r:
        st.markdown("**Rejected**")
        for _, row in rejected.iterrows():
            label = f"{row['flavor']}  ·  {float(row['opportunity_score']):.2f}  ·  {row['confidence']}"
            st.button(
                label,
                key=f"rej_{row['flavor']}",
                use_container_width=True,
                type="primary" if row["flavor"] == focus else "secondary",
                on_click=_select,
                args=(str(row["flavor"]),),
            )

    st.markdown("### Golden Candidate")
    score = float(golden.get("opportunity_score") or 0)
    st.markdown(
        f"""
        <div class="gc-wrap">
            <div class="fs-kicker" style="color:#E7C9A4;">Golden Candidate</div>
            <h2>{golden.get("flavor")}</h2>
            <div class="gc-score">{score:.2f} / 100</div>
            <div class="gc-line">Strongest flavor opportunity in the analyzed dataset.</div>
            <div class="gc-line">Based on the available consumer evidence - not a guaranteed commercial outcome.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Eligible mentions", int(golden.get("mentions") or 0))
    g2.metric("Positive sentiment", f"{float(golden.get('positive_rate') or 0) * 100:.1f}%")
    g3.metric("Brand-fit score", f"{float(golden.get('brand_fit_score') or 0):.2f}")
    g4.metric("Confidence", str(golden.get("confidence") or ""))
    st.markdown("**Why this works**")
    st.write(golden.get("why_it_works") or "")
    st.info(
        f"Recommended brand: **{golden.get('recommended_brand') or 'Needs validation'}**. "
        f"{golden.get('recommended_brand_rationale') or ''}"
    )
    with st.expander("HealthKart brand map (assignment context)"):
        st.write("MuscleBlaze → performance / gym / protein")
        st.write("HK Vitals → wellness / lifestyle")
        st.write("TrueBasics → premium / functional wellness")
        st.caption("A brand is assigned only when supporting SKUs clearly match that domain.")

    st.markdown("### Evidence")
    st.caption("Structured classifications from real Amazon reviews. Quotes are truncated source text.")
    try:
        evidence = _cached_evidence(focus)
    except Exception:
        evidence = pd.DataFrame()
    fallback = None
    if focus == str(golden.get("flavor")):
        fallback = (golden.get("evidence_summary") or {}).get("representative_reviews")
    evidence_cards(evidence, fallback)

    st.markdown("### Why you can trust this")
    st.write(HOW_IT_WORKS)
    chips = " <em>→</em> ".join(f"<span>{step}</span>" for step in PIPELINE_STEPS)
    st.markdown(f'<div class="pipe">{chips}</div>', unsafe_allow_html=True)
    st.caption("The LLM structures evidence. It does not invent the next flavor.")

    with st.expander("Data and methodology"):
        st.write(
            f"- Source: Amazon Reviews 2023, Sports & Outdoors (historical dataset, not live).\n"
            f"- Reviews analyzed: {int(data.stats.get('classified_reviews') or 0):,}\n"
            f"- Relevant reviews: {int(data.stats.get('relevant_reviews') or 0):,}\n"
            f"- Eligible flavor mentions: {int(data.stats.get('eligible_flavor_mentions') or 0):,}\n"
            f"- Scoring candidates: {int(data.stats.get('flavors_eligible_for_scoring') or 0)}\n"
            "- Purchase/request signals: 0\n"
            "- Reliable growth signal: unavailable\n"
            "- Opportunity score used Demand + Sentiment + Brand Fit because Purchase Intent and Growth were unavailable.\n"
            "- Recommendation is based on this dataset and is not a guaranteed commercial outcome."
        )
        st.caption(LAYOUT_RATIONALE)


main()
