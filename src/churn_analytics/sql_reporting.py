"""Evidence-backed SQL business report and automated reconciliation summary."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .reporting import markdown_table
from .sql_analytics import DEFAULT_PARAMETERS, QUERY_QUESTIONS, SEGMENT_QUERIES, read_query

INTERPRETATIONS = {
    "01_customer_kpis": "These metrics establish the customer denominator and outcome mix for every subsequent comparison. No means non-churned in this snapshot, not measured retention over a dated period.",
    "02_contract_segments": "Contract mix is a starting point for researching renewal experience and customer choices. A contract-offer experiment would be needed to estimate an intervention effect.",
    "03_tenure_segments": "The observed tenure differences motivate onboarding/cohort research. Acquisition and cancellation dates are needed for longitudinal analysis.",
    "04_monthly_charge_segments": "The fixed charge groups help compare plan/value experiences. Plan mix, tenure and service needs could explain differences.",
    "05_internet_segments": "Compare service experience within the recorded internet groups, then investigate incidents, pricing and customer needs.",
    "06_payment_method_segments": "Payment method may reflect billing friction or customer preferences. Failed-payment and complaint records would help investigate the association.",
    "07_technical_support_segments": "Support subscriptions may identify different customer needs and bundles. Contact history and service-quality evidence are needed before testing support interventions.",
    "08_online_security_segments": "Security subscriptions are descriptive attributes, not randomized treatments. Customer needs and plan bundles may explain the differences.",
    "09_senior_citizen_segments": "Use this comparison to investigate requested accessibility and service preferences, rather than assuming age itself determines outcomes.",
    "10_paperless_billing_segments": "The comparison can motivate investigation of communication preferences and billing clarity, with other customer attributes considered.",
    "11_service_combinations": "Joint segments show how service mix can affect aggregate comparisons. They help prioritize customer-experience research, not prescribe a proven churn remedy.",
    "12_contract_churn_contribution": "Rate measures within-group prevalence; contribution measures share of all churned customers. Population size matters when interpreting which groups account for the largest number of cases.",
    "13_adjacent_tenure_bands": "Cumulative customer counts show how much of the snapshot falls within shorter tenure bands; adjacent-rate differences compare distinct customer groups.",
    "14_substantial_above_average_segments": "These groups meet a transparent population-size condition and exceed the observed overall churn proportion, making them candidates for further investigation.",
    "15_charges_by_churn": "Mean and median describe different aspects of charge distributions. Pricing, bundle mix and tenure should be investigated before attributing outcomes to charges.",
    "16_customer_charge_positions": "Billing/plan review can compare a customer's recorded charge with the contract mean. These are price positions, not churn scores or predictions.",
    "17_monthly_charge_quartiles": "Equal-customer-count groups reveal distributional differences across charge levels; they complement the fixed Phase 2 business bands.",
    "18_contract_payment_mix": "Within-contract payment percentages show how payment mix differs across contracts, helping assess potential confounding in the aggregate payment comparison.",
    "19_missing_charge_coverage": "Coverage checks prevent treating unavailable amounts as actual zero charges and make observed-value denominators explicit.",
    "20_filtered_contract_tenure": "This bounded contract/tenure profile is useful for onboarding research and provides a meaningful equality-plus-range query for the indexing example.",
}
CAVEATS = {
    "01_customer_kpis": "No event dates support monthly rates. Charges are dataset charge units, not true revenue loss, CLV or ROI. The ordered-position median needs a sort.",
    "02_contract_segments": "Customers self-select into contracts; these associations do not prove longer contracts prevent churn. LEFT JOIN keeps dictionary categories even when n=0.",
    "03_tenure_segments": "Cross-sectional tenure differences are subject to survivor selection. Fixed group widths differ, and empty groups have undefined rates.",
    "04_monthly_charge_segments": "Bins <50, 50 to <80 and 80+ are fixed analytical choices, not optimized risk thresholds or price interventions.",
    "05_internet_segments": "No internet service is a genuine structural category. Differences do not establish service technology as a causal driver.",
    "06_payment_method_segments": "Payment choice is not randomized. This dictionary JOIN uses one source and is not multi-source integration.",
    "07_technical_support_segments": "No internet service is not equivalent to declining support; subscription status does not measure actual support quality.",
    "08_online_security_segments": "Structural non-subscribers and internet customers without security are distinct groups. No causal benefit is estimated.",
    "09_senior_citizen_segments": "Status is binary, not detailed age. Observational differences do not justify exclusionary customer actions.",
    "10_paperless_billing_segments": "Other billing, plan and demographic attributes may confound this unadjusted comparison.",
    "11_service_combinations": "Minimum n=100 is an analyst-defined descriptive filter, not proof of statistical reliability. Results depend on the threshold; smaller groups are excluded from this output only.",
    "12_contract_churn_contribution": "A high count can arise from a large population. Rankings are descriptive; tied rates share ranks, and zero total churn makes contributions undefined.",
    "13_adjacent_tenure_bands": "LAG/LEAD operate on ordered tenure bands, NOT months in historical time. Differences are percentage points, not temporal changes or hazards.",
    "14_substantial_above_average_segments": "Minimum n=100 and the overall sample rate are exploratory filters. No model, optimized threshold, significance claim or predicted risk is produced.",
    "15_charges_by_churn": "AVG(TotalCharges) excludes NULL; observed coverage is shown. Group differences do not estimate price effects or revenue loss.",
    "16_customer_charge_positions": "ROW_NUMBER uses customerID only to make tied-price selection deterministic; it is not a predictive feature. Top n does not include every tied customer at the cutoff.",
    "17_monthly_charge_quartiles": "NTILE splits rows, so equal charges may appear in adjacent tiles. These are whole-snapshot descriptive bins, not fitted preprocessing for future evaluation.",
    "18_contract_payment_mix": "The CROSS JOIN defines analytical cells only; it does not fabricate customers. Count matched IDs rather than LEFT JOIN placeholder rows.",
    "19_missing_charge_coverage": "Zero tenure does not establish a zero charge; source-system reasons for missingness remain unknown.",
    "20_filtered_contract_tenure": "A segment with no observed churn still has sampling uncertainty. Parameters are bound as values; percentages are undefined when no rows match.",
}


def result_markdown(frame: pd.DataFrame) -> str:
    """Show presentation-rounded results; CSV exports retain full numeric precision."""
    if frame.empty:
        return "_No rows meet the documented query parameters._"
    def format_value(value):
        if pd.isna(value):
            return "NULL / undefined"
        if isinstance(value, (float, np.floating)):
            return f"{value:.6f}".rstrip("0").rstrip(".")
        return str(value)
    return markdown_table(frame.map(format_value))


def _observation(name: str, frame: pd.DataFrame) -> str:
    """Derive a leading numerical observation directly from each returned result."""
    if frame.empty:
        return "No qualifying rows were returned for these parameters."
    if name in SEGMENT_QUERIES or name in {"11_service_combinations", "14_substantial_above_average_segments"}:
        known_keys = [SEGMENT_QUERIES[name]] if name in SEGMENT_QUERIES else (
            ["InternetService", "TechSupport", "OnlineSecurity"] if name.startswith("11") else ["Contract", "InternetService"])
        observed = frame.dropna(subset=["churn_pct"])
        if observed.empty:
            return "All returned segments are empty; their churn percentages are undefined."
        row = observed.loc[observed.churn_pct.idxmax()]
        label = ", ".join(f"{key}={row[key]}" for key in known_keys)
        return f'Highest rate among returned segments: **{label}**, **{row.churn_pct:.2f}%** ({int(row.churn_count):,}/{int(row.customer_count):,}).'
    if name == "12_contract_churn_contribution":
        row = frame.loc[frame.churn_count.idxmax()]
        return f'{row.Contract} accounts for **{int(row.churn_count):,} churned customers**, **{row.share_of_all_churn_pct:.2f}%** of all churned customers in this snapshot.'
    if name == "13_adjacent_tenure_bands":
        return f'The final cumulative customer count across the ordered bands is **{int(frame.iloc[-1].cumulative_customers):,}**.'
    if name == "17_monthly_charge_quartiles":
        return f'The quartile customer counts range from **{int(frame.customer_count.min()):,}** to **{int(frame.customer_count.max()):,}**.'
    return f'The query returned **{len(frame)}** result row(s); the complete calculated values are below.'


def reconciliation_markdown(comparison: pd.DataFrame) -> str:
    """Summarize all checks without hiding failures or altering expectations."""
    return "\n\n".join([
        "# Phase 3 SQL/Pandas reconciliation",
        f'Checks passed: **{int(comparison.passed.sum())}/{len(comparison)}**.',
        "Counts compare exactly. Floating values compare with absolute and relative tolerance 1e-10; matched undefined rates compare equal. Outer joins detect unmatched segment keys. SQL percentages are explicitly compared with 100 × Pandas proportions. No rounding is used for validation.",
        result_markdown(comparison),
    ]) + "\n"


def sql_business_report(results: dict, ingestion: dict, comparison: pd.DataFrame, performance: dict) -> str:
    """Answer every business question with SQL, actual output and interpretation."""
    k = results["01_customer_kpis"].to_dict("records")[0]
    counts = ingestion["table_rows"]
    sections = [
        "# Phase 3 — SQL customer analytics and business insights",
        "## Executive summary",
        f'The database contains **{k["total_customers"]:,} customers**, **{k["churn_count"]:,} churned** and **{k["retained_count"]:,} labeled No**. Overall sample churn is **{k["churn_pct"]:.2f}%**; the non-churned share is **{k["retention_pct"]:.2f}%**. Mean/median monthly charges are **{k["mean_monthly_charges"]:.2f}/{k["median_monthly_charges"]:.2f}** dataset charge units. All **{len(comparison)} SQL/Pandas checks** match. Results are descriptive associations, not proven retention effects.',
        "## Database overview and relationships",
        f'Input is the existing Phase 2 processed CSV, loaded without new cleaning. `customer_snapshot` has one row per unique customerID and all 25 analytical columns. Contract ({counts["contract_lookup"]} rows) and payment-method ({counts["payment_method_lookup"]} rows) lookups contain actual source categories; tenure ({counts["tenure_group_lookup"]}) and charge-group ({counts["monthly_charge_group_lookup"]}) lookups describe Phase 2’s fixed bins. These cardinalities come from the loaded artifact, not a fabricated customer history. Four many-to-one foreign keys enforce the relationships. See the checked-in Phase 3 guide for the ER diagram and complete data-model explanation.',
        f'Inserted customer rows: **{ingestion["inserted_records"]:,}**. SQL NULL cumulative charges: **{ingestion["null_total_charges"]}**. Integrity check: **{ingestion["integrity_check"]}**; foreign-key violations: **{ingestion["foreign_key_violations"]}**.',
        "SQLite >=3.37 STRICT tables use TEXT labels/IDs, INTEGER flags/counts and REAL charges. Original Pandas boolean flags map to checked 0/1 integers; ordered-category metadata lives in lookup display_order columns. Lookup joins organize one dataset, not complex multi-source integration.",
        "## Business questions and actual results",
        "SQL dialect: SQLite. Default parameter values are shown. All result tables are presentation-rounded to at most six decimals; unrounded CSVs are under `phase3/query_results/`. Percentages use each segment's own matched customer count. NULL rates are undefined, never zero.",
    ]
    for index, (name, question) in enumerate(QUERY_QUESTIONS.items(), 1):
        sections += [f"### {index}. {question}", f"Query: `sql/queries/{name}.sql`",
                     "```sql\n" + read_query(name).strip() + "\n```"]
        if name in DEFAULT_PARAMETERS:
            sections.append("Parameters: `" + json.dumps(DEFAULT_PARAMETERS[name], sort_keys=True) + "`.")
        sections += ["**Actual result:** " + _observation(name, results[name]), result_markdown(results[name]),
                     "**Business interpretation:** " + INTERPRETATIONS[name], "**Limitations:** " + CAVEATS[name]]
    before, after = performance["before_index"], performance["after_index"]
    timing = pd.DataFrame([
        {"Condition": "Without covering index", "Median ms": before["median_ms"], "Minimum ms": before["minimum_ms"], "p95 ms": before["p95_ms"], "Executions": before["repeats"]},
        {"Condition": "With covering index", "Median ms": after["median_ms"], "Minimum ms": after["minimum_ms"], "p95 ms": after["p95_ms"], "Executions": after["repeats"]},
    ])
    sections += ["## Query performance observations", result_markdown(timing),
                 "Benchmark query: #20 with its stated parameters. " + performance["method"],
                 "Before plan:\n```text\n" + "\n".join(r["detail"] for r in before["plan"]) + "\n```",
                 "After plan:\n```text\n" + "\n".join(r["detail"] for r in after["plan"]) + "\n```",
                 "Both executions returned matching results. These measured local timings are not a guaranteed speedup: the copy is memory-resident, warm and small; blocks are sequential; Python/preparation overhead is included, index construction/storage/write costs are excluded. Low-selectivity queries can favor a scan. Query-plan text can vary by SQLite version. No PostgreSQL server was installed or benchmarked.",
                 "## SQL/Pandas validation", f'**{int(comparison.passed.sum())}/{len(comparison)} checks passed**; full-customer round-trip validation also passed. See [automated reconciliation](phase3/sql_pandas_reconciliation.md) and its CSV/JSON companions. Counts require exact equality; unrounded floats use abs_tol=rel_tol=1e-10.',
                 "## Limitations and next analytical investigations",
                 "This snapshot has no transactions, dated subscription history or monthly activity. Churn counts/rates describe recorded outcomes; they do not establish causal effects, temporal changes, customer lifetime value, true revenue loss or retention ROI. Sampling and confounding limitations from Phase 2 remain. Minimum segment sizes and NTILE groups are descriptive choices; no predicted churn risk is produced.",
                 "Investigate onboarding experience, service incidents, plan/contract selection and billing complaints using genuine operational data. Evaluate retention hypotheses through credible controlled studies before claiming improvements or savings. Phase 4 requires an explicit new request; no model or dashboard is implemented here.",
                 "References: [SQLite STRICT tables](https://www.sqlite.org/stricttables.html), [window functions](https://www.sqlite.org/windowfunctions.html), [query-plan interpretation](https://www.sqlite.org/eqp.html). The checked-in guide also compares PostgreSQL using its official documentation."]
    return "\n\n".join(sections) + "\n"
