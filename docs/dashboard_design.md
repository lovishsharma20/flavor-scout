# Flavor Scout dashboard design

## 1. Target user

A HealthKart product manager who needs a defensible flavor recommendation, not a notebook of model metrics.

## 2. User decision journey

1. What are consumers talking about?
2. Which ideas are worth pursuing?
3. What is the single strongest opportunity?
4. Why, in evidence they can audit?
5. Can this be trusted, and what is missing?

## 3. Page flow

Market Pulse → Trend Wall → Decision Engine → Golden Candidate → Consumer Evidence → Trust / Methodology

This is a single scrolling page. There is no sidebar or navigation bar. The reviewer should move naturally from dataset size, to what was mentioned, to what is worth pursuing, to the #1 recommendation, to supporting reviews, then to limits and method.

## 4. Why Trend Wall comes first among the decision views

The assignment requires a Trend Wall. A ranked horizontal bar chart is the primary view so a manager can compare mention volume without decoding a word cloud. Market Pulse sits above it so the scale of the dataset is visible first.

## 5. Why Decision Engine comes next

After seeing the landscape, the user needs a binary business call: Selected vs Rejected. Thresholds are project rules, shown in methodology, not hidden in code.

## 6. Why Golden Candidate is visually dominant

The assignment asks for one Golden Candidate. That card uses the strongest contrast so the recommendation is impossible to miss, then immediately qualifies it with “why this works” and limits. Brand fit is shown as a numeric score. The page does not force a HealthKart brand name in the Golden Candidate block.

## 7. Why Consumer Evidence follows the recommendation

Auditability: the user should see real review text and structured fields (flavor, sentiment, intent, pain point, brand fit, confidence) after seeing the answer. The inspect control defaults to Strawberry.

## 8. Why limitations / methodology are included

Purchase intent is 0 and growth is unavailable. Showing those gaps is analytical maturity and prevents the dashboard from looking like a live forecasting tool.

## 9. How the UI avoids treating LLM output as unquestionable truth

The pipeline is printed as: real reviews → LLM classification → structured evidence → deterministic aggregation → score → Decision Engine → Golden Candidate. Copy states that the LLM structures evidence and does not invent the final flavor. Missing growth and purchase intent are labeled as unavailable, never filled with placeholder percentages.

Consumer Evidence shows only reviews that were eligible for aggregation for the inspected flavor, and excludes gear/container SKUs even if an older classification marked them relevant.
