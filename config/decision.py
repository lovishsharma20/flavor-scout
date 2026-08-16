"""
Project Decision Engine thresholds.

These are Flavor Scout product-team rules, not facts from Amazon Reviews 2023.
Change them here; do not hard-code them in the engine loop.
"""

# opportunity_scores.csv stores opportunity_score and brand_fit_score on a 0–100 scale.
# positive_rate remains a 0–1 share.
DECISION_OPPORTUNITY_SCORE_MIN = 50.0
DECISION_MENTIONS_MIN = 2
DECISION_POSITIVE_RATE_MIN = 0.50
DECISION_BRAND_FIT_SCORE_MIN = 50.0  # equivalent to 0.50 on the 0–1 brand-fit scale
DECISION_CONFIDENCE_ALLOWED = ("HIGH", "MEDIUM")
DECISION_LABEL_SELECTED = "SELECTED"
DECISION_LABEL_REJECTED = "REJECTED"
