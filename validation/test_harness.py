"""
The harness itself, exercised on the shared seed-42 runs: every claim it
tracks is evaluated, the report renders, and the claims that held in
EVERY seed of both scenarios when the standing report was generated
(reports/validation_report.md) also hold here. Claims that held in fewer
seeds are reported by the harness, not asserted, so the suite does not
encode luck.
"""

from __future__ import annotations

import pytest

from validation.harness import CLAIMS, evaluate_run, format_report

ROBUST_CLAIMS = (
    "naive_not_above_random_top10",
    "t_beats_naive_top10",
    "x_beats_naive_top10",
    "x_lower_rmse_than_t",
    "naive_policy_loses_money_10",
    "roi_policies_profitable_10",
    "roi_policies_beat_naive_all_budgets",
    "planned_lists_underspend",
    "model_valuation_overstates",
)


@pytest.mark.slow
@pytest.mark.parametrize("run_name", ["balanced_run", "imbalanced_run"])
def test_robust_claims_hold_and_report_renders(run_name, request):
    result = evaluate_run(request.getfixturevalue(run_name), n_bootstrap=50)
    assert set(result.claims) == set(CLAIMS)
    assert result.policy is not None  # harness runs carry economics + revenue truth
    for claim in ROBUST_CLAIMS:
        assert result.claims[claim], claim

    report = format_report([result], header="test")
    for description, _ in CLAIMS.values():
        assert description in report
