"""
Reusable, rule-based cleaning helpers for Flavor Scout Reddit text.
"""

from __future__ import annotations

import re
import unicodedata

# --- thresholds / patterns (easy to tune) ---

MIN_CHARS = 20
MIN_WORDS = 4

DELETED_MARKERS = {
    "[deleted]",
    "[removed]",
    "deleted",
    "removed",
}

# Topic signals we want to keep (flavor / product discussion)
RELEVANCE_KEYWORDS = [
    "flavor",
    "flavour",
    "taste",
    "tastes",
    "tasting",
    "protein",
    "whey",
    "casein",
    "supplement",
    "electrolyte",
    "electrolytes",
    "shake",
    "powder",
    "scoop",
    "vanilla",
    "chocolate",
    "strawberry",
    "banana",
    "cookie",
    "cinnamon",
    "mocha",
    "caramel",
    "peanut",
    "berry",
    "fruit",
    "sweet",
    "bitter",
    "chalky",
    "artificial",
    "aftertaste",
    "recommend",
    "recommendation",
    "prefer",
    "preference",
    "compare",
    "comparison",
    "vs",
    "versus",
    "better than",
    "worse than",
    "dislike",
    "hate the taste",
    "love the taste",
    "too sweet",
    "not sweet",
    "drink mix",
    "preworkout",
    "pre-workout",
    "bcaa",
    "creatine",
    "collagen",
]

SPAM_PATTERNS = [
    r"\buse\s+(my\s+)?code\b",
    r"\bpromo\s*code\b",
    r"\bdiscount\s*code\b",
    r"\baffiliate\b",
    r"\bfree\s+shipping\b",
    r"\bbuy\s+now\b",
    r"\border\s+now\b",
    r"\bclick\s+(here|the\s+link)\b",
    r"\bcheck\s+out\s+my\b",
    r"\bdm\s+me\b",
    r"\blink\s+in\s+(bio|profile)\b",
    r"\bsubscribe\s+to\s+(my|our)\b",
    r"\bfollow\s+me\b",
    r"https?://\S*(shop|store|amazon|amzn\.to|bit\.ly|tinyurl)\S*",
]

BOT_PATTERNS = [
    r"\bi\s+am\s+a\s+bot\b",
    r"\bthis\s+action\s+was\s+performed\s+automatically\b",
    r"\bautomoderator\b",
    r"\bbot\s+message\b",
    r"\bplease\s+contact\s+the\s+moderators\b",
    r"^good\s+bot[.!]*$",
    r"^bad\s+bot[.!]*$",
]

EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F9FF"  # symbols & pictographs / supplemental
    "\U0001FA00-\U0001FAFF"
    "\U00002700-\U000027BF"  # dingbats
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport
    "\U00002600-\U000026FF"  # misc symbols
    "]+",
    flags=re.UNICODE,
)

URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, strip URLs/emoji noise lightly, collapse whitespace."""
    if text is None:
        return ""
    value = unicodedata.normalize("NFKC", str(text)).strip().lower()
    value = URL_RE.sub(" ", value)
    value = WHITESPACE_RE.sub(" ", value).strip()
    return value


def is_empty_or_deleted(text: str) -> bool:
    cleaned = normalize_text(text)
    return not cleaned or cleaned in DELETED_MARKERS


def is_too_short(text: str, min_chars: int = MIN_CHARS, min_words: int = MIN_WORDS) -> bool:
    cleaned = normalize_text(text)
    if len(cleaned) < min_chars:
        return True
    return len(cleaned.split()) < min_words


def is_emoji_only(text: str, max_non_emoji_chars: int = 3) -> bool:
    """True when almost nothing remains after removing emoji/punctuation."""
    if text is None:
        return True
    raw = str(text).strip()
    if not raw:
        return True
    without_emoji = EMOJI_RE.sub("", raw)
    # keep letters/digits only
    leftover = re.sub(r"[^\w]+", "", without_emoji, flags=re.UNICODE)
    return len(leftover) <= max_non_emoji_chars


def is_spam(text: str) -> bool:
    cleaned = normalize_text(text)
    return any(re.search(pattern, cleaned, flags=re.IGNORECASE) for pattern in SPAM_PATTERNS)


def is_bot(text: str) -> bool:
    cleaned = normalize_text(text)
    return any(re.search(pattern, cleaned, flags=re.IGNORECASE) for pattern in BOT_PATTERNS)


def is_irrelevant(text: str) -> bool:
    """Drop rows with no flavor/product-related signal words."""
    cleaned = normalize_text(text)
    return not any(keyword in cleaned for keyword in RELEVANCE_KEYWORDS)


def duplicate_key(text: str) -> str:
    """Normalized key used for near-exact duplicate detection."""
    return normalize_text(text)
