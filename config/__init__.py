"""Configurable settings for Flavor Scout MVP."""

# --- Step 1: Amazon Reviews 2023 Sports_and_Outdoors (official raw file) ---
# Historical dataset (through Sep 2023), not live Amazon data.
# Docs: https://amazon-reviews-2023.github.io/
AMAZON_CATEGORY = "Sports_and_Outdoors"
AMAZON_REVIEW_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
    "raw/review_categories/Sports_and_Outdoors.jsonl.gz"
)
# Official item metadata for the same category. Join key is parent_asin.
AMAZON_META_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
    "raw/meta_categories/meta_Sports_and_Outdoors.jsonl.gz"
)

# Product/category signals for ingestion (NOT LLM relevance).
# Flavor/taste words are intentionally omitted so later LLM can classify requests,
# complaints, and experiences that never mention "flavor".
AMAZON_REVIEW_KEYWORDS = [
    "protein",
    "whey",
    "casein",
    "isolate",
    "supplement",
    "electrolyte",
    "hydration",
    "sports nutrition",
    "pre-workout",
    "preworkout",
    "pre workout",
    "post-workout",
    "post workout",
    "recovery drink",
    "bcaa",
    "amino acid",
    "creatine",
    "collagen",
    "protein powder",
    "protein shake",
    "gummy",
    "gummies",
    "muscleblaze",
    "hk vitals",
    "truebasics",
]

# Internship MVP target: stop once this many unique relevant reviews are kept.
AMAZON_TARGET_REVIEWS = 8000

# --- Step 3: LLM classification (pilot = 50 reviews) ---
CLASSIFY_PILOT_LIMIT = 50
CLASSIFY_LIMIT = 50
CLASSIFY_MODEL = "gpt-4o-mini"
CLASSIFY_MAX_RETRIES = 3
CLASSIFY_SLEEP_SECONDS = 0.2
