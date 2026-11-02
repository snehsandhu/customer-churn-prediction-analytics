"""Known-answer tests for proportions, segment denominators and statistics."""
import math

import numpy as np
import pandas as pd
import pytest

from churn_analytics.analytics import (
    association_tests, categorical_summary, churn_by_segment, correlation_tables,
    customer_kpis, customer_segment_counts, holm_adjust, numerical_summary, wilson_interval,
)


@pytest.fixture
def known_customers():
    """Four customers with hand-computable KPIs and unequal segment sizes."""
    return pd.DataFrame({
        "ChurnFlag": [1, 0, 0, 1], "Churn": ["Yes", "No", "No", "Yes"],
        "Segment": pd.Categorical(["A", "A", "A", "B"], categories=["A", "B", "Empty"]),
        "MonthlyCharges": [10.0, 20.0, 30.0, 40.0], "tenure": [0, 10, 20, 30],
        "TotalCharges": [np.nan, 100.0, 200.0, 300.0],
    })


def test_kpis_have_known_numerators_and_units(known_customers):
    k = customer_kpis(known_customers)
    assert (k["total_customers"], k["churn_count"], k["retained_count"]) == (4, 2, 2)
    assert k["churn_proportion"] == k["retention_proportion"] == 0.5
    assert k["mean_monthly_charges"] == k["median_monthly_charges"] == 25
    assert k["mean_tenure_months"] == k["median_tenure_months"] == 15
    assert k["mean_total_charges_observed"] == 200
    assert k["total_charges_observed_customers"] == 3


def test_segment_denominators_and_empty_segment(known_customers):
    result = churn_by_segment(known_customers, "Segment").set_index("Segment")
    assert result.loc["A", "customer_count"] == 3
    assert result.loc["A", "churn_rate"] == pytest.approx(1 / 3)
    assert result.loc["A", "churn_pct"] == pytest.approx(100 / 3)
    assert result.loc["B", "churn_rate"] == 1
    assert result.loc["Empty", "customer_count"] == 0
    assert math.isnan(result.loc["Empty", "churn_rate"])
    assert math.isnan(result.loc["Empty", "ci_low"])
    assert result.customer_count.sum() == len(known_customers)
    counts = customer_segment_counts(known_customers, "Segment").set_index("Segment")
    assert counts.loc["A", "customer_proportion"] == 0.75


def test_multivariate_denominators_reconcile(cleaned_customers):
    result = churn_by_segment(cleaned_customers, ["Contract", "TenureGroup"])
    assert result.customer_count.sum() == len(cleaned_customers)
    assert result.churn_count.sum() == cleaned_customers.ChurnFlag.sum()
    empty = result.loc[result.customer_count.eq(0)]
    assert empty.churn_rate.isna().all()


def test_empty_dataset_is_undefined_not_zero(known_customers):
    empty = known_customers.iloc[:0]
    k = customer_kpis(empty)
    assert k["total_customers"] == 0
    assert math.isnan(k["churn_proportion"])
    assert churn_by_segment(empty, "Segment").churn_rate.isna().all()


def test_invalid_outcomes_cannot_silently_change_denominator(known_customers):
    known_customers.loc[0, "ChurnFlag"] = 2
    with pytest.raises(ValueError, match="ChurnFlag"):
        customer_kpis(known_customers)
    with pytest.raises(ValueError, match="ChurnFlag"):
        churn_by_segment(known_customers, "Segment")


def test_wilson_known_reference_and_boundaries():
    # NIST-style Wilson score formula for 50 successes in 100 trials.
    assert wilson_interval(50, 100) == pytest.approx((0.4038315303659957, 0.5961684696340044))
    assert wilson_interval(0, 10)[0] == pytest.approx(0, abs=1e-15)
    assert wilson_interval(10, 10)[1] == pytest.approx(1)
    assert all(math.isnan(x) for x in wilson_interval(0, 0))
    for args in [(2, 1), (-1, 10), (1, -1), (1.5, 10)]:
        with pytest.raises(ValueError):
            wilson_interval(*args)


def test_holm_known_answer_preserves_input_order():
    np.testing.assert_allclose(holm_adjust([0.04, 0.001, 0.02]), [0.04, 0.003, 0.04])
    np.testing.assert_allclose(holm_adjust([0.9, 0.8]), [1, 1])
    with pytest.raises(ValueError):
        holm_adjust([np.nan])


def test_chi_square_known_table_and_effect_size():
    # Observed [[30,10],[10,30]]; expected all 20, chi-square 20, V=0.5.
    data = pd.DataFrame({"Segment": ["A"] * 40 + ["B"] * 40,
                         "ChurnFlag": [0] * 30 + [1] * 10 + [0] * 10 + [1] * 30})
    result = association_tests(data, ["Segment"]).iloc[0]
    assert result.chi2 == pytest.approx(20)
    assert result.cramers_v == pytest.approx(0.5)
    assert result.min_expected == 20
    assert result.p_value == pytest.approx(7.744216431044088e-6)
    assert result.supports_association


def test_sparse_or_degenerate_tests_do_not_claim_support():
    sparse = pd.DataFrame({"Segment": ["A", "B"], "ChurnFlag": [0, 1]})
    result = association_tests(sparse, ["Segment"]).iloc[0]
    assert not result.assumptions_ok
    assert not result.supports_association
    degenerate = sparse.assign(ChurnFlag=0)
    result = association_tests(degenerate, ["Segment"]).iloc[0]
    assert math.isnan(result.p_value)
    assert not result.supports_association


def test_numeric_statistics_and_pairwise_coverage(known_customers):
    summary = numerical_summary(known_customers).set_index("feature")
    assert summary.loc["MonthlyCharges", "q1"] == 17.5
    assert summary.loc["MonthlyCharges", "q3"] == 32.5
    assert summary.loc["MonthlyCharges", "iqr"] == 15
    assert summary.loc["TotalCharges", "missing_count"] == 1
    corr = correlation_tables(known_customers)
    assert list(corr["pearson"].columns) == ["tenure", "MonthlyCharges", "TotalCharges"]
    assert corr["pair_counts"].loc["tenure", "TotalCharges"] == 3
    assert corr["pair_counts"].loc["tenure", "MonthlyCharges"] == 4
    assert corr["spearman"].loc["tenure", "TotalCharges"] == pytest.approx(1)


def test_categorical_counts_reconcile_for_each_feature(cleaned_customers):
    table = categorical_summary(cleaned_customers)
    assert table.groupby("feature").customer_count.sum().eq(8).all()
    assert table.groupby("feature").customer_proportion.sum().eq(1).all()
