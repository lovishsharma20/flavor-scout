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

LLM_PROVIDER = "groq"
OPENAI_MODEL = "gpt-4o-mini"

OPENAI_MODEL_BENCHMARK = "gpt-4.1-mini"
OPENAI_INPUT_PRICE_PER_MILLION = 0.40
OPENAI_OUTPUT_PRICE_PER_MILLION = 1.60

OPENAI_TPM_LIMIT = 200_000

GROQ_MODEL = "llama-3.3-70b-versatile"

GROQ_MODEL_FULL = "llama-3.1-8b-instant"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
CLASSIFY_MODEL = GROQ_MODEL
CLASSIFY_PILOT_LIMIT = 50
CLASSIFY_LIMIT = 50
CLASSIFY_MAX_RETRIES = 5
CLASSIFY_SLEEP_SECONDS = 0.4
CLASSIFY_FULL_SLEEP_SECONDS = 2.1
CLASSIFY_CHECKPOINT_EVERY = 10
CLASSIFY_PILOT_SEED = 42

CLASSIFY_BATCH_SIZE = 4
CLASSIFY_TPM_LIMIT = 6000
CLASSIFY_TPM_BUDGET_FRACTION = 0.75
CLASSIFY_MAX_PROMPT_TOKENS_PER_BATCH = 3200
CLASSIFY_BATCH_BENCHMARK_LIMIT = 100

CLASSIFY_COST_STOP_USD = 4.50
CLASSIFY_OPENAI_BATCH_SIZE_TEST = 8

CLASSIFY_OPENAI_MAX_PROMPT_TOKENS_PER_BATCH = 10000
CLASSIFY_ASYNC_CONCURRENCY = 3
