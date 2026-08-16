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

LAYOUT_RATIONALE = (
    "This dashboard follows a product manager's decision journey: first see "
    "what consumers are talking about, then which ideas are worth pursuing, "
    "then the strongest opportunity, then evidence and methodology so the "
    "recommendation is auditable."
)
