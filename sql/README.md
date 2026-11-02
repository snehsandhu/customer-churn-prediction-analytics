# SQL analytics — SQLite dialect

Run from the existing project root with its Python 3.13 environment activated:

```bash
python -m churn_analytics.phase3
python -m churn_analytics.phase3 --query 02_contract_segments
python -m churn_analytics.phase3 --query 11_service_combinations --params '{"min_customers":500}'
python -m churn_analytics.phase3 --query 20_filtered_contract_tenure --params '{"contract":"Two year","min_tenure":0,"max_tenure":12}'
```

The first command validates and reads the **existing processed Phase 2 CSV**,
refreshes the managed SQLite snapshot transactionally, executes every query,
reconciles SQL/Pandas, and generates reports. Named-query commands open the
existing database read-only; they do not rerun ingestion or benchmarks. Unknown
query names and parameter names are rejected; values are bound, never interpolated.

`schema.sql` defines five STRICT tables. `indexes.sql` defines two deliberate
indexes. `queries/` contains twenty separate analytical SELECT statements. SQLite
>=3.37 is required for STRICT tables and supports the window functions used here.
No third-party database package or PostgreSQL installation is required.

| Query | Business purpose / main SQL concepts |
| --- | --- |
| 01 | Snapshot KPIs; conditional aggregation, scalar subquery, ordered-position median |
| 02 | Contract churn; LEFT JOIN, matched-ID count, safe denominator |
| 03 | Tenure-band churn; lookup ordering, empty bands |
| 04 | Fixed charge-group churn; lookup ordering |
| 05 | Internet-service churn; GROUP BY |
| 06 | Payment-method churn; actual dictionary INNER JOIN |
| 07 | Technical-support churn; structural service categories |
| 08 | Online-security churn; within-segment percentages |
| 09 | SeniorCitizen category comparison |
| 10 | Paperless-billing category comparison |
| 11 | Joint service segments; GROUP BY/HAVING with bound minimum n |
| 12 | Contract rate vs volume vs contribution; CTE, RANK, DENSE_RANK, window SUM |
| 13 | Adjacent tenure-band comparison; LAG, LEAD, cumulative window SUM |
| 14 | Substantial groups above overall churn; CTE/HAVING plus overall-rate subquery |
| 15 | Charges by outcome; partitioned ordered-position median and NULL coverage |
| 16 | Top charges within contract; PARTITION BY, ROW_NUMBER, RANK, DENSE_RANK, window AVG |
| 17 | Customer-count charge quartiles; NTILE |
| 18 | Payment mix within contract; CROSS/LEFT JOIN, aggregation then window denominator |
| 19 | Missing-charge coverage; CASE, conditional aggregation, COUNT(column) |
| 20 | Selected contract/tenure profile; named parameters, equality/range filtering, covering-index benchmark |

Percentages use `100.0` to avoid integer truncation and `NULLIF(denominator,0)`
to return NULL for undefined rates. `COUNT(customerID)` counts matched customers
after a LEFT JOIN; `COUNT(*)` would count an unmatched placeholder row. Financial
NULLs remain NULL. Source labels, not arbitrary encoded categories, drive groups.

`LAG`/`LEAD` compare ordered **cross-sectional tenure bands**, not historical
calendar changes. NTILE makes equal-sized row groups and can split tied charges;
it is distinct from the fixed Phase 2 charge labels. Neither segmentation produces
predictions or estimates causal retention effects.

See `docs/PHASE3_SQL_GUIDE.md` for the ER diagram, step-by-step query walkthroughs,
index interpretation, PostgreSQL comparison, file map and interview questions.
Generated results: `reports/phase3/query_results/`. Every question, full SQL text,
actual results, interpretation and caveat appears in
`reports/PHASE3_SQL_BUSINESS_INSIGHTS.md`.
