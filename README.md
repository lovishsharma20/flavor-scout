# Flavor Scout

Consumer intelligence for identifying evidence-backed flavor opportunities for HealthKart (MuscleBlaze, HK Vitals, TrueBasics).

This is a completed interview MVP. Analysis is already finished offline. The dashboard does **not** call an LLM and is **not** a live Amazon feed.

## Problem

Consumer reviews contain useful flavor preferences, complaints, and pain points, but the text is unstructured. A product team cannot turn thousands of reviews into a launch decision without a clear evidence trail.

## Solution

Flavor Scout uses an LLM to structure **review-level** evidence from real Amazon reviews. Deterministic aggregation, scoring, and decision rules then identify the strongest opportunity. The model does not invent demand, growth, purchase intent, or the Golden Candidate.

## Pipeline

Amazon Reviews → Cleaning → LLM Classification → Flavor QC → Trend Aggregation → Opportunity Scoring → Decision Engine → Golden Candidate → Streamlit Dashboard

| Stage | Module |
|---|---|
| Ingestion | `src/ingestion/amazon_ingest.py` |
| Cleaning + metadata | `src/cleaning/clean_reviews.py` |
| LLM classification | `src/classification/classify_reviews.py` |
| Flavor QC + trends | `src/aggregation/flavor_qc.py`, `trend_engine.py` |
| Opportunity Score | `src/aggregation/opportunity_score.py` |
| Decision Engine | `src/aggregation/decision_engine.py` |
| Golden Candidate | `src/aggregation/golden_candidate.py` |
| Dashboard | `app.py`, `src/dashboard/` |

## Dataset

Source: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (McAuley Lab), category `Sports_and_Outdoors`. Historical reviews through September 2023 — not live Amazon data.

Ingestion streams the official Sports & Outdoors `.jsonl.gz` and keeps a manageable sports-nutrition sample. Cleaning joins item metadata and applies rule-based filters (no LLM).

Completed analysis:

- **7,931** cleaned reviews classified
- **486** relevant reviews
- **50** eligible flavor mentions
- **11** scoring-eligible flavors (at least 2 mentions)
- **4** selected opportunities
- **7** rejected opportunities

## AI classification

The LLM classifies **each review**, not the dataset as a whole. For every review it returns:

- relevance
- flavor
- sentiment
- intent
- pain point
- brand fit (`strong` / `moderate` / `weak` / `none`)
- reasoning
- confidence

The LLM structures evidence from real review text. It does **not** choose the Golden Candidate and does not compute the Opportunity Score.

The full 7,931-review run used OpenAI `gpt-4.1-mini`. Groq remains available as an alternative provider. Re-running classification is not required to use the dashboard.

## Anti-hallucination architecture

Real consumer reviews → LLM structured evidence → deterministic aggregation → deterministic scoring → deterministic Decision Engine → Golden Candidate

Underlying consumer data is the source of truth. Missing signals are labeled unavailable. They are not filled with placeholder percentages.

## Opportunity Score

The assignment discusses Demand, Growth, Sentiment, Purchase Intent, and Brand Fit. In **this** dataset:

- purchase/request signals among eligible mentions = **0**
- reliable flavor-level temporal growth = **unavailable**

Those signals were **not fabricated** and were **excluded** from the numerical score. The remaining 65 weight points were renormalized:

- Demand: 30 / 65
- Sentiment: 20 / 65
- Brand Fit: 15 / 65

Demand is min-max normalized mention volume among the 11 scoring candidates (2–9 mentions). Sentiment is the positive share × 100. Brand fit is the 0–1 aggregation score scaled to 0–100.

## Final result

**Golden Candidate: Strawberry — 91.45 / 100**

Highest Opportunity Score among Decision Engine SELECTED flavors. 9 eligible mentions, 88.9% positive sentiment, brand-fit 77.78, HIGH confidence.

Current evidence is mostly freeze-dried fruit / fruit powder in Sports & Outdoors. That is not enough to assign MuscleBlaze, HK Vitals, or TrueBasics without further category validation. The dashboard reports brand-fit as a score; it does not force a brand name.

**SELECTED**

- Strawberry
- Chocolate
- Lemonade
- Banana

**REJECTED**

- Raspberry Lime
- Coconut
- Blueberry
- Orange
- Cranberry
- Lemon Lime
- Birthday Cake

Missing purchase intent and growth were **not** used as automatic reject reasons.

## Limitations

- Source is historical Amazon Reviews 2023, not a live social feed.
- Sports & Outdoors mixes supplements with gear; LLM relevance plus SKU/gear QC reduce that noise.
- Only 486 of 7,931 reviews were relevant to flavor discovery.
- Eligible flavor volume is small (50 mentions).
- Zero purchase/request signals in the eligible sample.
- No reliable growth signal; a stored Strawberry time split was not used as trend growth.
- The recommendation is evidence-based for this sample, not a guaranteed commercial outcome.

## Dashboard

The Streamlit app is a presentation layer over completed analysis files. It does not call OpenAI or Groq.

Product-manager journey (one page, top to bottom):

1. **Market Pulse** — how large the analyzed dataset is
2. **Trend Wall** — what consumers in this dataset mentioned
3. **Decision Engine** — what is worth pursuing (selected vs rejected)
4. **Golden Candidate** — the strongest opportunity
5. **Consumer Evidence** — supporting reviews for the inspected flavor
6. **Trust / methodology** — how the AI is used, and what is missing

### Why this layout?

A product manager should see the landscape first, then the business call, then the single recommendation, then auditable evidence, then limitations. That matches “what are consumers talking about → what is worth pursuing → what is #1 → why → can I trust it?”

## Run locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501

The dashboard needs the completed files already in `data/processed/` (`flavor_trends.csv`, `opportunity_scores.csv`, `decision_engine_results.csv`, `golden_candidate.json`). It does **not** need an API key.

To reproduce ingestion or classification from scratch, copy `.env.example` to `.env` and add keys. Do not paste keys into chat or commit `.env`.

## Hosted dashboard

A Streamlit Community Cloud deployment is planned for the assignment deliverable. It is **not** live yet. A clean GitHub checkout can run the dashboard locally from the tracked analysis files. Full review-level evidence cards also use local `analyzed_reviews.csv` and `flavor_mention_qc.csv` (gitignored because of size). If those files are absent, Strawberry still shows stored excerpts from `golden_candidate.json`.

## License / use

Interview project for a Data Analyst Intern role. Not an official HealthKart product.
