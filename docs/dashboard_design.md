# Flavor Scout dashboard design

## 1. Target user

A HealthKart product manager who needs a defensible flavor recommendation, not a notebook of model metrics.

## 2. User decision journey

1. What are consumers talking about?
2. Which ideas are worth pursuing?
3. What is the single strongest opportunity?
4. Why, in evidence they can audit?
5. Can this be trusted, and what is missing?

## 3. Why Trend Wall comes first

The assignment requires a Trend Wall. A ranked horizontal bar chart is the primary view so a manager can compare mention volume without decoding a word cloud.

## 4. Why Decision Engine comes second

After seeing the landscape, the user needs a binary business call: Selected vs Rejected. Thresholds are project rules, shown in methodology, not hidden in code.

## 5. Why Golden Candidate is visually dominant

The assignment asks for one Golden Candidate. That card uses the strongest contrast so the recommendation is impossible to miss, then immediately qualifies it with evidence and limits.

## 6. Why Evidence follows the recommendation

Auditability: the user should see real review text and the structured LLM fields (relevance, flavor, sentiment, intent, pain point, brand fit) after seeing the answer.

## 7. Why limitations / methodology are included

Purchase intent is 0 and growth is unavailable. Showing those gaps is analytical maturity and prevents the dashboard from looking like a live forecasting tool.

## 8. How the UI avoids treating LLM output as unquestionable truth

The pipeline is printed as: real reviews → LLM classification → structured evidence → deterministic aggregation → score → Decision Engine → Golden Candidate. Copy states that the LLM structures evidence and does not invent the final flavor. Missing growth and purchase intent are labeled as unavailable, never filled with placeholder percentages.
