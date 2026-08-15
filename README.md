# Flavor Scout

Interview project for a Data Analyst Intern role. Discovers flavor opportunities in protein, whey, supplements, electrolytes, and sports nutrition using **historical** Amazon Reviews 2023 consumer reviews.

> This project does **not** use live Amazon data. Reviews come from the public McAuley Lab Amazon Reviews 2023 dataset (interactions through September 2023).

## Pipeline (planned)

1. Amazon Dataset Ingestion
2. **Metadata enrichment + cleaning** ← current
3. LLM Classification
4. Flavor Aggregation
5. Opportunity Scoring
6. Selected / Rejected (Decision Engine)
7. Golden Candidate
8. Streamlit UI (Trend Wall)
9. Deployment

## Dataset source (Step 1)

- **Name:** Amazon Reviews 2023 (McAuley Lab)
- **Docs:** https://amazon-reviews-2023.github.io/
- **Category:** `Sports_and_Outdoors`
- **Official review file (streamed, not saved locally):**
  - https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Sports_and_Outdoors.jsonl.gz
- **Access method:** stream remote `.jsonl.gz` over HTTP, filter relevant reviews in memory, write only the extracted CSV

### Official review fields (schema)

`rating`, `title`, `text`, `asin`, `parent_asin`, `user_id`, `timestamp`, `verified_purchase`, `helpful_vote`

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
# Edit .env — add OPENAI_API_KEY before classification
```

`.env` is gitignored. See `.env.example` for the template.

## Step 1 — Amazon review ingestion

### Probe connection / schema (first 1,000 records only)

```bash
python -m src.ingestion.amazon_ingest --probe 1000
```

### Full extraction (only after approval)

```bash
python -m src.ingestion.amazon_ingest --full
```

Output: `data/raw/amazon_reviews.csv`

Tune keywords in `config/__init__.py` (`AMAZON_REVIEW_KEYWORDS`).

## Step 2 — Metadata enrichment + cleaning

Joins official Sports & Outdoors **item metadata** on `parent_asin` (streamed; the full metadata file is not stored). Then applies rule-based cleaning only (no LLM).

```bash
python -m src.cleaning.clean_reviews
```

- Input (unchanged): `data/raw/amazon_reviews.csv`
- Output: `data/processed/processed_reviews.csv`

Latest run (historical Amazon Reviews 2023 sample):

- Starting rows: 8,000
- Final rows: 7,931 (99.1% retained)
- Duplicates removed: 26
- Spam/promotional removed: 7
- Unusable noise removed: 36
- Irrelevant removed: 0 (LLM relevance is Step 3)
- Metadata: all 5,155 `parent_asin` values matched; every processed row has `product_title`
- Brand present on 7,859 of 7,931 processed rows

Limitation: ingestion used product/category keywords such as `hydration`, so some non-supplement products (for example hydration backpacks) remain. Step 3 LLM classification is where finer relevance filtering happens.

## Step 3 — LLM classification (50-review pilot)

Uses OpenAI + Pydantic. Does **not** classify the full 7,931 rows.

```bash
python -m src.classification.classify_reviews
```

Requires `OPENAI_API_KEY` in `.env` (a real key from https://platform.openai.com/api-keys, not the `your_...` placeholder).

Output: `data/processed/analyzed_reviews_pilot.csv`
