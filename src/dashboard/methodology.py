"""Copy for methodology, limitations, and anti-hallucination."""

PIPELINE_STEPS = [
    "Real consumer reviews",
    "LLM classification",
    "Structured evidence",
    "Deterministic trend aggregation",
    "Deterministic opportunity score",
    "Decision Engine",
    "Golden Candidate",
]

HOW_IT_WORKS = (
    "The LLM analyzes and structures real consumer reviews. It does not invent "
    "the final flavor recommendation. Trend aggregation, scoring, selection, "
    "and Golden Candidate logic are deterministic. Underlying consumer data "
    "is the source of truth."
)
