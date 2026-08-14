"""Configurable settings for Flavor Scout MVP."""

# --- Legacy Reddit settings (kept; Amazon is now Step 1 source) ---
TARGET_TOTAL = 80
POSTS_PER_QUERY = 15
COMMENTS_PER_POST = 5
SORT = "relevance"  # relevance | hot | top | new | comments
TIME_FILTER = "year"  # all | year | month | week | day | hour

SUBREDDITS = [
    "supplements",
    "Fitness",
    "nutrition",
    "wheyprotein",
    "bodybuilding",
    "HomemadeProteinBars",
    "Electrolytes",
]

QUERIES = [
    "protein powder flavor",
    "whey protein taste",
    "best tasting whey",
    "protein shake flavor preference",
    "electrolyte drink flavor",
    "supplement flavor recommendations",
    "chocolate vs vanilla protein",
    "protein powder too sweet",
]

# --- Step 1: Amazon Reviews 2023 Sports_and_Outdoors (official raw file) ---
# Historical dataset (through Sep 2023), not live Amazon data.
# Docs: https://amazon-reviews-2023.github.io/
AMAZON_CATEGORY = "Sports_and_Outdoors"
AMAZON_REVIEW_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
    "raw/review_categories/Sports_and_Outdoors.jsonl.gz"
)

# Keyword filter applied to review title + text while streaming
AMAZON_REVIEW_KEYWORDS = [
    "protein",
    "whey",
    "supplement",
    "electrolyte",
    "bcaa",
    "pre-workout",
    "preworkout",
    "pre workout",
    "creatine",
    "nutrition",
    "flavor",
    "flavour",
    "taste",
]

AMAZON_MIN_REVIEW_CHARS = 30

# --- LLM classification (later step; unchanged) ---
CLASSIFY_LIMIT = 30
CLASSIFY_MODEL = "gpt-4o-mini"
CLASSIFY_MAX_RETRIES = 3
CLASSIFY_SLEEP_SECONDS = 0.2
