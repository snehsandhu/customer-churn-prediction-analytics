> Publication-copy note: this guide documents the historical complete local
> project. Trusted datasets/models/provenance and private audit logs are withheld
> from this source release. See [asset requirements](PUBLIC_RELEASE_ASSETS.md),
> [current verification](PUBLICATION_VERIFICATION.md) and
> [new authentic captures](screenshots/README.md). Historical development commands
> are for review; do not retrain frozen models or reopen the consumed test.

# Phase 2 learning guide

This guide explains the implemented workflow and prepares you to discuss its
choices in a Data Analyst interview. The generated business report contains the
actual numerical evidence; `PHASE2_VERIFICATION.md` records executed checks.

## New files and their responsibilities

| New file | Responsibility |
| --- | --- |
| `src/churn_analytics/cleaning.py` | Category/range/service rules, deterministic cleaning, before/after quality assessment, protected CSV save, typed reload |
| `src/churn_analytics/analytics.py` | Snapshot KPIs, segment counts/rates, numerical/categorical summaries, Wilson intervals, Holm-adjusted exploratory tests, correlations |
| `src/churn_analytics/plots.py` | Consistent Matplotlib/Seaborn charts, labels, complete-case coverage and segment-size annotations |
| `src/churn_analytics/reporting.py` | Readable quality/business Markdown reports generated from computed evidence |
| `src/churn_analytics/phase2.py` | One-command orchestration, CSV round-trip check, raw-file fingerprint and manifest |
| `notebooks/02_data_cleaning_eda.ipynb` | Teaching narrative and executable cleaning, EDA, visualizations and observations |
| `tests/conftest.py` | Shared small, valid synthetic test records; no network or fabricated portfolio evidence |
| `tests/test_cleaning.py` | Cleaning, invalid values, invariants, CSV reload and raw-file preservation tests |
| `tests/test_analytics.py` | Hand-computable KPI, denominator, missing-data, interval and association-test checks |
| `tests/test_phase2.py` | Offline end-to-end CSV/report generation and repeat-run checks; figure rendering is replaced in these fast unit tests |
| `docs/PHASE2_GUIDE.md` | This explanation, file map and interview questions |
| `PHASE2_VERIFICATION.md` | Actual executed commands, results, findings and limitations |

Existing `README.md` is extended rather than replaced. `requirements.txt` now
explicitly declares SciPy, previously installed transitively through scikit-learn.
`pyproject.toml` declares the numerical/statistical/plotting packages used by the
package. Phase 1's `data.py`, tests, notebook and verification record are retained.

### Generated artifacts

| Path | Contents |
| --- | --- |
| `data/processed/telco_churn_cleaned.csv` | Validated analytical dataset: 7,043 records and 25 columns for the verified IBM original |
| `reports/data_quality_report.json` | Full before/after types, missingness, categories, ranges, duplication, effects and alternative-policy sensitivity |
| `reports/data_quality_report.md` | Human-readable before/after audit |
| `reports/kpis.json` | Customer counts, sample churn/non-churn proportions, charge/tenure summaries and coverage |
| `reports/tables/*.csv` | 29 summaries: numerical/categorical distributions, churn segments, three multivariate tables, subscriber-only security, eight association tests and three correlation tables |
| `reports/figures/*.png` | 17 Matplotlib/Seaborn figures listed below |
| `reports/PHASE2_BUSINESS_INSIGHTS.md` | Executive summary, measured evidence, hypotheses and limitations |
| `reports/phase2_manifest.json` | Source/output fingerprints, generated paths, package versions and KPIs |
| `reports/02_data_cleaning_eda.executed.ipynb` | Executed notebook copy, preserving a clear-output source notebook |
| `reports/phase2_environment.json` | Package versions recorded during final verification |
| `reports/phase2_chart_review_*.png` | Temporary visual QA contact sheets; not project deliverables |

Processed data and generated artifacts remain ignored by Git, following Phase 1.
Run the workflow after cloning to regenerate them. No GitHub repository is created
or published here.

## The cleaning workflow and decisions

1. Load the unchanged original through Phase 1's `load_dataset()`. Capture its
   SHA-256 fingerprint for a before/after comparison.
2. Require the expected source schema (or a complete already-cleaned schema).
   Work on a deep copy. Do not silently discard extra/missing columns or records.
3. Strip outer whitespace from string fields. Retain case and original category
   meaning; unexpected labels are errors, not automatically guessed corrections.
4. Parse numeric columns with `errors="raise"`. Convert empty/whitespace-only
   numeric text to missing first. Malformed nonblank numbers remain errors.
5. Validate customer identity, target values, category domains, finite nonnegative
   numerical values, whole-number tenure, SeniorCitizen=0/1, and service
   consistency. For example, No internet service must agree with every add-on's
   structural label. Missing values outside TotalCharges are rejected.
6. Retain missing TotalCharges only at zero tenure. Any new missing charge at
   nonzero tenure raises an error so this narrowly justified policy is revisited.
7. Add four deterministic analytical columns, then validate them against source
   values. Re-running recomputes them and yields the same DataFrame.
8. Save a separate CSV atomically under `data/processed/`, reload with documented
   types and revalidate. Never save into `data/raw/`. Every analytical output uses
   the reloaded artifact, and the source fingerprint is checked again.

**Missing versus invalid:** a blank charge means its value is unavailable. A
nonblank value like `"not a number"`, an infinite charge, a negative tenure, an
unknown target or a contradictory service label violates a documented rule.
Missingness requires a policy; invalid values require investigation. Neither is
silently changed to zero.

### Why retain the 11 missing charges?

Actual inspection shows that all 11 records have zero tenure and Churn=No.
A new-account billing cycle or an incomplete extract are possible explanations,
not verified facts. No field establishes the actual charge. The chosen policy
preserves all 7,043 customers, retains missingness as NaN and adds an explicit
flag. There is no learned imputation and no assumption that the charge is zero.

| Strategy | Advantages | Trade-offs in this actual dataset |
| --- | --- | --- |
| Retain missing — chosen | Preserve all customers and uncertainty | Charge summaries require observed-value denominators; TotalCharges has 7,032 observations |
| Impute zero — not applied | Produces a complete numeric field | Assumes an undocumented financial value; lowers mean TotalCharges from 2,283.30 to 2,279.73 |
| Exclude rows — not applied | Complete TotalCharges for remaining records | Removes 11 zero-tenure, non-churned customers; sample churn changes from 26.54% to 26.58% |

All alternative numbers are calculated by `build_quality_report()`; only the
chosen dataset is saved. Other valid extreme observations are retained because an
IQR flag alone is not proof of an error. We do not recalculate TotalCharges as
MonthlyCharges × tenure, because no rule establishes that equality for each record.

### Derived-column dictionary

| Column | Meaning and boundaries |
| --- | --- |
| `ChurnFlag` | int8: Yes=1, No=0. This is the target in numeric form, never a predictive feature |
| `TotalChargesMissing` | Boolean flag matching `TotalCharges.isna()` |
| `TenureGroup` | Ordered fixed bins: 0–12, 13–24, 25–36, 37–48, 49–60, 61+ months |
| `MonthlyChargeGroup` | Ordered fixed bins: <50, 50 to <80, 80+ in dataset charge units; exactly 50 belongs to the middle bin and exactly 80 to the upper bin |

Original columns retain their names and semantics. SeniorCitizen and tenure are
integers; monthly/total charges are floats; original text fields are strings.
CSV does not retain category metadata, so `load_cleaned_dataset()` restores and
checks ordered groups. A blank processed TotalCharges field represents missing
information and is reloaded as NaN. Both Churn and ChurnFlag are outcome columns;
neither, nor customerID, belongs in future predictive features.

## EDA, groupby and KPI definitions

**Univariate** means one variable: a tenure histogram, a monthly-charge median or
contract frequencies. **Bivariate** means two variables: churn by contract,
charges by churn or tenure versus cumulative charges. **Multivariate** means
several variables together: churn within contract × tenure segments.

For a binary target, the mean equals its proportion of ones:

```python
from churn_analytics.analytics import customer_kpis, churn_by_segment

kpis = customer_kpis(cleaned)
contracts = churn_by_segment(cleaned, "Contract")
services = churn_by_segment(cleaned, "TechSupport")
combined = churn_by_segment(cleaned, ["Contract", "TenureGroup"])

# Conceptual calculation underlying a segment's result:
cleaned.groupby("Contract")["ChurnFlag"].agg(["size", "sum", "mean"])
```

`size` is the number of customers in that segment. `sum` counts churned customers.
`mean = sum / size`; multiply by 100 for a percentage. Do not divide each segment's
churn count by all 7,043 customers when claiming a within-segment rate. Empty
segments return NaN rates and intervals, not misleading zeros. The production
helper includes unobserved categorical combinations and flags n<30.

Total customers is the row count after identity/quality checks. Churn count means
Churn=Yes; retained count means Churn=No in the snapshot. Churn and retention
proportions sum to one but are not measured monthly operational rates. The mean
and median monthly charges are per-customer charge statistics, not revenue loss.
Mean/median cumulative charges exclude missing charges and report coverage.

## Statistics and chart choices

Numerical summaries include observed/missing counts, mean, median, sample standard
deviation (ddof=1), min/max, quartiles, IQR and 1.5×IQR outlier counts. Histograms
show distributions; class-normalized histograms compare shapes despite unequal
class sizes; boxplots compare medians/spread without deleting fliers. Frequency
plots show categorical sizes. Churn bars show within-segment percentages on a
0–100% scale, with counts and 95% Wilson intervals. Scatterplots show every complete
pair. Multivariate heatmaps display rate and n, masking empty combinations.

The numerical correlation matrix includes only tenure and charges. Pearson
summarizes linear association, Spearman monotonic rank association. Pairwise
counts reveal missing-value exclusions. No arbitrary category codes, IDs or
outcome-derived variables enter the correlation matrix.

Eight fixed feature-level Pearson chi-square tests are exploratory. Holm adjusts
for multiple testing within this family. Cramér's V provides an uncorrected
association magnitude. Tests need every observed/expected cell ≥5 to receive the
support flag. Wilson intervals and tests rely on sampling assumptions the public
snapshot cannot verify; they do not establish representativeness or causality.
Intervals are marginal and their overlap is not used as a significance test.

Method references: [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html),
[NIST Wilson intervals](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm),
[Holm correction](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html).

### Chart filenames

- `churn_distribution.png`
- `churn_rate_by_contract.png`
- `tenure_distribution_by_churn.png`
- `monthly_charges_distribution_by_churn.png`
- `churn_rate_by_internet_service.png`
- `churn_rate_by_payment_method.png`
- `churn_by_selected_services.png` (also includes categorical SeniorCitizen)
- `numerical_correlation_heatmap.png`
- `tenure_vs_total_charges.png`
- `churn_rate_by_tenure_group.png`
- `monthly_charges_vs_total_charges.png`
- `numerical_histograms.png`
- `numerical_boxplots_by_churn.png`
- `categorical_frequency_counts.png`
- `churn_contract_tenure.png`
- `churn_internet_technical_support.png`
- `churn_contract_monthly_charges.png`

## Measured findings and their limits

Sample churn is 1,869 / 7,043 = 26.54%; the No proportion is 73.46%.
Mean/median monthly charges are 64.76/70.35 dataset charge units.
Month-to-month churn is 1,655 / 3,875 = 42.71%, compared with 48 / 1,695 = 2.83%
for two-year contracts. Tenure 0–12 months has 1,037 / 2,186 = 47.44%, versus
93 / 1,407 = 6.61% for 61+ months. Electronic-check customers have
1,071 / 2,365 = 45.29%; fiber-optic customers have 1,297 / 3,096 = 41.89%.
Mean monthly charges are 74.44 among churned and 61.27 among non-churned customers.

These differences motivate research into onboarding, plan choice, billing and
service experience. They do not prove that changing contracts, payment methods,
prices or support will improve retention. Need dated events, customer reasons,
operational evidence and controlled experiments before recommending causal action
or estimating savings. No monthly trends, CLV or retention ROI are computed.
See the generated report for sample sizes, intervals and evidence for every major
finding, plus suggestions for further investigation.

Full-dataset EDA is exploratory. For later modeling, reserve evaluation data before
fitting learned transformations or choosing models/thresholds; never use the final
test set to make those choices. Since this phase inspects the full data,
acknowledge that future evaluation data has been seen descriptively and seek
unseen data for independent confirmation. No models are implemented in Phase 2.

## Twenty interview questions with answer pointers

1. **Why return a copy from a cleaning function?** Avoid modifying the caller's
   raw DataFrame; reproducibility tests compare the original before and after.
2. **What is deterministic, idempotent cleaning?** Same input produces the same
   output, and cleaning an already-cleaned frame produces the same result.
3. **How does Pandas distinguish whitespace from NaN?** Spaces are non-null text;
   use string stripping/fullmatch as well as `isna()`.
4. **Why not replace the 11 missing charges with zero?** Zero tenure provides
   context but no documented actual financial value. Preserve uncertainty.
5. **How would removing those rows affect this sample?** It removes 11 No labels
   and changes sample churn from 26.54% to 26.58%.
6. **What is the difference between missing and invalid values?** Missing is
   unavailable information; invalid violates a parsing, range or domain rule.
7. **Why use `pd.to_numeric(errors="raise")` here?** Reject malformed nonblank
   values rather than silently turning them into missing information.
8. **Why keep No internet service separate from No?** One means the underlying
   service is absent; the other means an available add-on is not subscribed to.
9. **Why check both duplicate rows and duplicate IDs?** Exact duplication and
   conflicting records for the same identity are different quality problems.
10. **What do univariate, bivariate and multivariate mean?** One variable, two
    variables, and combinations of several variables; give notebook examples.
11. **How does `groupby().mean()` calculate churn rates?** A 0/1 mean is the churn
    count divided by that group's customer count.
12. **Why is the denominator important?** Contract sizes differ; 1,655/3,875 and
    48/1,695 are comparable proportions, unlike raw churn counts alone.
13. **What does an empty segment's churn rate mean?** It is undefined, not zero;
    return NaN and visibly label no observations.
14. **Why show segment sizes with percentages?** Rates from small groups can be
    unstable; flag n<30 and inspect uncertainty rather than overinterpret.
15. **How do median, IQR and boxplots help?** Summarize central tendency/spread
    without assuming symmetry; IQR outliers warrant review, not automatic deletion.
16. **Why use normalized histograms by churn?** Compare distribution shapes when
    the No and Yes groups have different sizes; counts are shown separately.
17. **Why report pair counts with correlations?** TotalCharges has only 7,032
    observations, so those correlations use fewer rows than other numeric pairs.
18. **What do chi-square, Holm and Cramér's V each add?** Independence assessment,
    multiple-test adjustment and association magnitude; none establishes causality.
19. **Can you conclude a two-year contract prevents churn?** No. The observed
    2.83% versus 42.71% is confounded by customer selection and other attributes.
20. **What would you need to estimate trends or retention ROI?** Dated events,
    consistent populations, actual costs/outcomes and credible intervention-effect
    evidence; the snapshot alone is insufficient.
