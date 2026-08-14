# Flavor Scout

Interview project for a Data Analyst Intern role. Discovers flavor opportunities in protein, whey, supplements, electrolytes, and sports nutrition using **historical** Amazon Reviews 2023 consumer reviews.

> This project does **not** use live Amazon data. Reviews come from the public McAuley Lab Amazon Reviews 2023 dataset (interactions through September 2023).

## Pipeline (planned)

1. **Amazon Dataset Ingestion** ← current
2. Cleaning
3. LLM Classification
4. Flavor Aggregation
5. Opportunity Scoring
6. Selected / Rejected
7. Golden Candidate
8. Streamlit UI
9. Deployment

## Dataset source (Step 1)

- **Name:** Amazon Reviews 2023 (McAuley Lab)
- **Docs:** https://amazon-reviews-2023.github.io/
- **Category:** `Sports_and_Outdoors`
- **Official review file (streamed, not saved locally):**
  - https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Sports_and_Outdoors.jsonl.gz
- **Access method:** stream remote `.jsonl.gz` over HTTP, filter relevant reviews in memory, write only the small extracted CSV

### Official review fields (schema)

`rating`, `title`, `text`, `asin`, `parent_asin`, `user_id`, `timestamp`, `verified_purchase`, `helpful_vote`  
(Some dumps also expose `helpful_votes` / `sort_timestamp`; the ingest script accepts both.)

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

Amazon ingestion needs `pandas` only (no Amazon API key). OpenAI is only for later classification.

## Step 1 — Amazon review ingestion

### 1) Probe connection / schema (first 1,000 records only)

```bash
python -m src.ingestion.amazon_ingest --probe 1000
```

### 2) Full extraction (only after approval)

```bash
python -m src.ingestion.amazon_ingest --full
```

Output: `data/raw/amazon_reviews.csv`

Tune keywords in `config/__init__.py` (`AMAZON_REVIEW_KEYWORDS`).

## Later steps (not migrated to Amazon yet)

Cleaning / classification still expect the earlier Reddit-shaped files. Do not run them on Amazon output until those steps are updated.
