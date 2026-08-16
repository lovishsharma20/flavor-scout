"""Configurable settings for Flavor Scout MVP."""

AMAZON_CATEGORY = "Sports_and_Outdoors"
AMAZON_REVIEW_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
    "raw/review_categories/Sports_and_Outdoors.jsonl.gz"
)
AMAZON_META_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
    "raw/meta_categories/meta_Sports_and_Outdoors.jsonl.gz"
)

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

AMAZON_TARGET_REVIEWS = 8000

CLASSIFY_PILOT_LIMIT = 50
CLASSIFY_LIMIT = 50
CLASSIFY_MODEL = "gpt-4o-mini"
CLASSIFY_MAX_RETRIES = 3
CLASSIFY_SLEEP_SECONDS = 0.2
