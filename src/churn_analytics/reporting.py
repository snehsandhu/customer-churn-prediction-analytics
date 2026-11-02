"""Render calculated evidence as readable Markdown without optional table packages."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .analytics import churn_by_segment, numerical_summary


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small DataFrame while preserving explicit missing-data markers."""
    def cell(value):
        if pd.isna(value):
            return "undefined"
        if isinstance(value, (float, np.floating)):
            return f"{value:.3g}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    rows = ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def segment_evidence(frame: pd.DataFrame, features: str | list[str]) -> str:
    """Format segment numerators, denominators, rates and uncertainty bounds."""
    keys = [features] if isinstance(features, str) else features
    table = churn_by_segment(frame, keys)
    result = table[[*keys, "customer_count", "churn_count", "small_segment"]].rename(
        columns={"customer_count": "Customers", "churn_count": "Churned", "small_segment": "n < 30"})
    result["Churn (%)"] = table["churn_pct"].map(lambda n: f"{n:.2f}" if pd.notna(n) else "undefined")
    result["95% Wilson interval (%)"] = [
        f"{100 * lo:.2f}–{100 * hi:.2f}" if pd.notna(lo) else "undefined"
        for lo, hi in zip(table.ci_low, table.ci_high)
    ]
    return markdown_table(result)


def quality_markdown(report: dict) -> str:
    """Show before/after types, missingness, duplication and the cleaning policy."""
    before, after = report["original"], report["cleaned"]
    rows = []
    for column, dtype in after["data_types"].items():
        rows.append({"Column": column, "Original dtype": before["data_types"].get(column, "new"),
                     "Clean dtype": dtype, "Original nulls": before["missing_values"].get(column, "—"),
                     "Original blanks": before["blank_values"].get(column, "—"),
                     "Clean nulls": after["missing_values"][column]})
    summary = pd.DataFrame([
        {"Check": "Rows", "Original": before["dimensions"]["rows"], "Cleaned": after["dimensions"]["rows"]},
        {"Check": "Columns", "Original": before["dimensions"]["columns"], "Cleaned": after["dimensions"]["columns"]},
        {"Check": "Duplicate IDs beyond first", "Original": before["duplicate_customer_ids"], "Cleaned": after["duplicate_customer_ids"]},
        {"Check": "Exact duplicate records", "Original": before["duplicate_records"], "Cleaned": after["duplicate_records"]},
    ])
    return "\n\n".join([
        "# Phase 2 data-quality report", markdown_table(summary),
        "## Policy", report["policy"],
        "No rows, valid outliers or structural service categories are removed. Outer whitespace is stripped; invalid numeric values, invalid categories, missing non-charge fields and service contradictions raise errors. Charge units are not assumed to represent revenue.",
        "## Types and missingness", markdown_table(pd.DataFrame(rows)),
        "## Alternative-policy sensitivity (hypothetical alternatives, not saved datasets)",
        markdown_table(pd.DataFrame(report["alternatives"])),
        "Alternative churn_rate values are proportions. Mean charges exclude missing values for the chosen policy, with the observed denominator shown. Zero imputation changes the charge mean without evidence that the true missing charges are zero.",
        "## Validation", f'Cleaned errors: {after["errors"]}. Warnings: {after["warnings"]}.',
        "Full numeric ranges, category values, whitespace counts and source consistency checks are in `data_quality_report.json`.",
    ]) + "\n"


def business_insights_markdown(frame: pd.DataFrame, kpis: dict, quality: dict, tests: pd.DataFrame) -> str:
    """Generate every quantitative statement directly from the analytical data."""
    contract = churn_by_segment(frame, "Contract").set_index("Contract")
    monthly = numerical_summary(frame, by_churn=True).query("feature == 'MonthlyCharges'").set_index("Churn")
    internet = frame.loc[frame["InternetService"].ne("No")]
    automatic = frame["PaymentMethod"].isin(["Bank transfer (automatic)", "Credit card (automatic)"])
    auto_n, auto_k = int(automatic.sum()), int(frame.loc[automatic, "ChurnFlag"].sum())
    auto_rate = auto_k / auto_n if auto_n else np.nan
    monthly_evidence = markdown_table(monthly.reset_index()[["Churn", "observed_count", "mean", "median", "std"]].round(2).astype(str))
    test_display = tests[["feature", "chi2", "degrees_freedom", "min_observed", "min_expected", "p_value", "p_value_holm", "cramers_v", "supports_association"]].copy()
    for c in ("p_value", "p_value_holm"):
        test_display[c] = test_display[c].map(lambda p: "undefined" if pd.isna(p) else ("<1e-300 (underflow)" if p == 0 else f"{p:.3g}"))
    correlation = frame[["tenure", "MonthlyCharges", "TotalCharges"]].corr()
    gap = None
    if {"Month-to-month", "Two year"}.issubset(contract.index):
        gap = 100 * (contract.loc["Month-to-month", "churn_rate"] - contract.loc["Two year", "churn_rate"])
    contract_note = f"Month-to-month minus two-year churn differs by {gap:.2f} percentage points." if gap is not None else "Compare only observed segments."
    return "\n\n".join([
        "# Phase 2 — Customer churn: descriptive business insights",
        "## Executive summary",
        f'This IBM snapshot contains **{kpis["total_customers"]:,} customers**, **{kpis["churn_count"]:,} churned** and **{kpis["retained_count"]:,} labeled No**. Sample churn is **{kpis["churn_proportion"]:.2%}**, and the non-churned proportion is **{kpis["retention_proportion"]:.2%}**. The overall 95% Wilson interval is {kpis["churn_ci_low"]:.2%}–{kpis["churn_ci_high"]:.2%}, conditional on a binomial sampling model. This is not a measured monthly business churn rate.',
        "Contract, tenure, service and payment segments are useful descriptive starting points. The comparisons below describe associations, not proven causes or estimated intervention effects. Operational priorities need customer context, costs, consent and experimental evidence.",
        "## Dataset and analysis scope",
        "Source: [IBM Telco Customer Churn CSV](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv). One row represents a customer record. Churn=Yes is coded 1 for arithmetic; IDs remain identifiers. Only cleaning and exploratory analysis are performed. There are no model predictions or holdout evaluation results. Charge statistics use dataset charge units without assuming a currency or equating charges to revenue loss.",
        "## Data-quality findings",
        f'All {len(frame):,} records and 21 source columns are retained; four documented analytical columns produce {frame.shape[1]} columns. Original duplicate IDs: {quality["original"]["duplicate_customer_ids"]}; exact duplicate records: {quality["original"]["duplicate_records"]}. Cleaned validation errors: {len(quality["cleaned"]["errors"])}. Numeric ranges, categorical domains and phone/internet service consistency are checked.',
        f'{kpis["total_charges_missing_customers"]} TotalCharges values remain missing, all at zero tenure. The source contains spaces rather than conventional nulls. They may reflect new-account billing timing or a source-system omission; this dataset cannot distinguish those explanations. No zero charge is assumed. Total-charge summaries use {kpis["total_charges_observed_customers"]:,} observed values and explicitly report coverage.',
        "See [data-quality report](data_quality_report.md) for before/after types, counts and comparisons with hypothetical zero-imputation or row-exclusion policies.",
        "## 1. Contract type and churn",
        segment_evidence(frame, "Contract"), contract_note,
        "**Why it matters:** contract segments distinguish customer groups with different observed churn prevalence. **Hypothesis to test:** clarity of renewal terms or a voluntary contract offer could be evaluated. **Further evidence needed:** renewal dates, plan pricing, reasons for leaving and a randomized offer evaluation; selection into contracts may explain part of the association.",
        "## 2. Tenure and churn",
        segment_evidence(frame, "TenureGroup"),
        "**Why it matters:** short-tenure customers may have different onboarding needs. **Hypothesis to test:** an onboarding check-in could improve the experience. **Further evidence needed:** acquisition cohorts, event dates, cancellation reasons and an experiment. Longer observed tenure is affected by survivor selection; this snapshot cannot show churn falling over calendar time or estimate a survival hazard.",
        "## 3. Monthly charges and cumulative charges",
        monthly_evidence,
        f'Overall monthly charges: mean **{kpis["mean_monthly_charges"]:.2f}**, median **{kpis["median_monthly_charges"]:.2f}**. Observed TotalCharges: mean **{kpis["mean_total_charges_observed"]:.2f}**, median **{kpis["median_total_charges_observed"]:.2f}**. Pearson correlations: tenure–TotalCharges **{correlation.loc["tenure", "TotalCharges"]:.3f}**; MonthlyCharges–TotalCharges **{correlation.loc["MonthlyCharges", "TotalCharges"]:.3f}**, using {kpis["total_charges_observed_customers"]:,} pairs. The saved correlation tables also include Spearman correlations and pair counts.',
        "**Why it matters:** charges can reflect plan mix, service bundles and tenure. **Hypothesis to test:** customers may benefit from clearer plan/value explanations. **Further evidence needed:** actual plans, promotions, billing disputes and price-change experiments. Different means and correlations do not identify a causal price effect; no monetary loss or saving is calculated.",
        "## 4. Internet service, support and security",
        segment_evidence(frame, "InternetService"),
        "Support within internet-service strata (structural combinations without observations have undefined rates):",
        segment_evidence(frame, ["InternetService", "TechSupport"]),
        "Online security among internet subscribers only:", segment_evidence(internet, "OnlineSecurity"),
        "**Why it matters:** these comparisons can guide investigation of customer experience and service bundles. **Hypothesis to test:** better support access or information about security could be evaluated. **Further evidence needed:** incidents, support contact history, customer needs and randomized interventions. A support subscription does not measure actual support quality, and subscribers self-select. Preserve No internet service as a structural category; it is not equivalent to declining an available add-on.",
        "## 5. Payment method and churn",
        segment_evidence(frame, "PaymentMethod"),
        f'Combined automatic-payment categories: {auto_k:,}/{auto_n:,} churned (**{auto_rate:.2%}**). This is a descriptive regrouping, not an additional significance test.',
        "**Why it matters:** payment method may proxy customer preferences, billing friction or other characteristics. **Hypothesis to test:** clearer billing or optional payment assistance could be evaluated. **Further evidence needed:** failed payments, billing complaints and an experiment. Moving a customer to automatic payment is not demonstrated to reduce churn.",
        "## 6. Senior-citizen status",
        segment_evidence(frame, "SeniorCitizen"),
        "SeniorCitizen is interpreted categorically (0/1), not as a continuous age measure. **Why it matters:** differing customer needs may warrant research. **Hypothesis to test:** accessible communication could help customers who request it. **Further evidence needed:** stated preferences and a service-design evaluation. This association alone is not a reason to target or exclude customers based on age.",
        "## Selected multivariate comparisons",
        "The corresponding heatmaps show contract × tenure, internet × technical support, and contract × fixed monthly-charge group. All segment tables include n, churn counts, rates and intervals. Groups below n=30 are marked; empty segments remain undefined. These stratifications do not adjust for every confounder and do not estimate treatment effects.",
        "See [contract × tenure](tables/segment_contract_tenure.csv), [internet × support](tables/segment_internet_support.csv), and [contract × monthly charges](tables/segment_contract_monthly_charges.csv).",
        "## Exploratory statistical support", markdown_table(test_display),
        "Pearson chi-square tests assess marginal independence for this fixed family of eight features. Holm adjustment controls family-wise error within that family under the test assumptions. Cramér’s V reports uncorrected association magnitude. A supports_association=True flag requires adjusted p<0.05 and every observed/expected cell count ≥5. Very small p-values do not measure business impact. These exploratory results are not confirmatory evidence for a new population or for post-hoc pairwise contrasts.",
        "The 95% Wilson intervals are marginal model-based intervals, not simultaneous intervals or a test based on visual overlap. The public snapshot does not establish independent random sampling, population representativeness, or the absence of confounding. No category search, model selection or threshold tuning is performed.",
        "Methods: [SciPy chi-square documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html), [NIST Wilson intervals](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm), [statsmodels description of Holm correction](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html). Holm is implemented directly; statsmodels is not a dependency.",
        "## Suggested investigations and limitations",
        "Validate these hypotheses on an appropriately sampled, dated customer population. Gather acquisition/cancellation dates, plans, service events and reasons for leaving. Evaluate interventions with controlled experiments and measured costs/outcomes before estimating ROI. No historical monthly trends, lifetime value, true revenue loss, causality or retention improvements can be inferred from this snapshot.",
        "Full-dataset EDA is exploratory. Later machine-learning work must create and lock evaluation splits before fitting learned transformations, choosing models or thresholds. A test set drawn after reviewing the full data has already been seen descriptively; document that limitation and prefer genuinely unseen data for independent confirmation. Do not fit preprocessing on this analytical dataset before splitting.",
        "All tables and [figures](figures/) are reproducible with `python -m churn_analytics.phase2`. No later project phase is implemented.",
    ]) + "\n"
