### The question

A cashback incentive is paid on **every treated conversion**, including people who would
have bought anyway. So the right question is not *"who will convert?"* but *"who converts
because of the cashback, and is that worth what we pay?"*

### The four kinds of customer (simulated; hidden from every model)

| Segment | Converts without cashback | Effect of cashback |
|---|---|---|
| Persuadable | rarely | large lift: the people the incentive is for |
| Sure thing | often | almost none, yet they collect the cashback |
| Lost cause | almost never | none |
| Sleeping dog | sometimes | negative: the offer puts them off |

### How a policy is built

1. **Models estimate uplift** τ(x): the change in conversion probability caused by treatment.
   The naive baseline instead predicts P(convert), which is the industry default.
2. **Shared economics** estimate each user's P(convert | treated) and revenue per conversion,
   so every policy is costed the same way.
3. **Rank by ROI** = τ / (rate × P(convert | treated)), then target down the list until the
   budget is spent. This is the greedy order for a spend cap.
4. **Value it honestly.** A model's own estimate over the users it picked is inflated by that
   very selection. The unbiased value comes from the randomized holdout: within the targeted
   users, compare treated and control revenue.

### What the validation harness found

Across 5 seeds × 2 scenarios (see `reports/validation_report.md`):

- **Naive cashback targeting loses money** at every tested budget. It is worse than random,
  because it pays sure-things to do what they would have done anyway.
- **ROI-ranked uplift policies are profitable** and beat naive at every budget.
- **Models overstate their own policies about 4×.** Trust the outcome-based estimate, but read
  its interval: on a ~10k-user holdout it is too wide to separate close policies.
- **Uplift models win at the top of the ranking, not across all of it.** Over the whole
  ranking, propensity tracks spend, and spend is a decent proxy for uplift. Which model is
  "best" depends on the budget, and that is exactly what this page lets you vary.
