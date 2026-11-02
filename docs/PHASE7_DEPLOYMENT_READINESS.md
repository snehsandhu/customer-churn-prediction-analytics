> Publication-copy note: this guide documents the historical complete local
> project. Trusted datasets/models/provenance and private audit logs are withheld
> from this source release. See [asset requirements](PUBLIC_RELEASE_ASSETS.md),
> [current verification](PUBLICATION_VERIFICATION.md) and
> [new authentic captures](screenshots/README.md). Historical development commands
> are for review; do not retrain frozen models or reopen the consumed test.

# Deployment and GitHub readiness — assessment, not deployment

**Local demonstration: functional. Public hosting: conditional and not performed.**
No repository, remote service, public URL, infrastructure, commit or deployment was
created. No package was installed/upgraded. These steps require owner approval.

## Current environment and prerequisites

Use the existing Python 3.13.9 environment. `requirements.txt` describes package
ranges; `requirements-demo-verified.txt` records the direct versions actually
verified on macOS. The latter is a reference, not a complete transitive or
cross-platform lock. Existing-environment dependency checking passed. A fresh
pip installation, Linux/Windows run and hosting-provider build are **UNVERIFIED**.

The loader requires the recorded scikit-learn 1.9.1 and verifies artifact identity.
Do not disable that guard, replace the model or retrain to accommodate a provider.
If compatible wheels/runtime cannot be supplied, stop and propose an explicitly
approved compatible environment; do not silently upgrade/downgrade dependencies.

Required application assets include:

- Phase 2 processed CSV and its Phase 4 source-fingerprint manifest.
- Phase 3 SQLite database and checked-in SQL definitions.
- Selected frozen RF, final selection manifest/model metadata/consumption ledger.
- Saved final aggregate metrics and validation comparison/importance/calibration tables.
- Saved model-performance figures used by the dashboard.

A source-only clone does not contain these ignored assets. The private review ZIP
includes them, without individual final-test predictions. It is not an approved
public release or an automatically available retrieval endpoint.

## Measured asset sizes

| Asset | Bytes |
|---|---:|
| Original IBM CSV | 970,457 |
| Processed CSV | 1,155,435 |
| SQLite database | 1,888,256 |
| Preserved LR artifact | 31,202 |
| RF baseline | 3,452,091 |
| RF balanced baseline | 4,835,924 |
| Selected tuned RF | 476,869 |

These sizes are small for local review. They do not establish peak RAM, startup
latency, concurrent-user capacity or compliance with a provider's current quota.
The QA server was launched locally; no production load/startup benchmark was run.

## Readiness checklist

| Check | Status / evidence |
|---|---|
| Local Streamlit entry and six pages | PASS — full AppTest plus limited actual Chrome review |
| Frozen model, exact schema and 0.50 threshold | PASS — service/tests/hash checks |
| Consumed test boundary | PASS — no Phase 7 evaluation or selection |
| SQL readonly, allowlists, exports | PASS — service/SQL and regression checks |
| Upload bounds | PASS — 1 MiB, 500 rows; synthetic validation/order checks |
| Relative asset paths | PASS — out-of-working-directory regression |
| Current dependency graph | PASS — existing packages only; no broken requirements |
| Fresh dependency resolver/build | UNVERIFIED — requires approval to install |
| Final desktop/tablet/mobile browser evidence | PARTIAL — capture tooling lost the Chrome window; final widths not verified |
| Linux/Windows/provider compatibility | UNVERIFIED |
| Artifact distribution/owner license | PENDING owner review; see attribution/distribution guide |
| Public authentication/remote privacy/load test | NOT IMPLEMENTED / NOT VERIFIED |
| Cloud/public deployment | NOT PERFORMED; explicit approval required |

## Hosting approach and approvals

A managed Streamlit host is a possible future option after compatibility and
artifact-distribution review. Community Cloud accepts a repository/entrypoint and
runtime selection; use the appropriate compatible runtime rather than assuming a
default. Check actual available wheels/settings at approval time.
[Official deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

Our `.streamlit/config.toml` binds **127.0.0.1** for local safety. A host may require
a provider-specific interface/port override. Such an override is a future approved
deployment action, not a reason to weaken the local defaults now. Do not copy this
private archive wholesale into a public repository.

After explicit approval, the owner/developer would:

1. Choose the project license and approve a reviewed source/distribution manifest.
2. Resolve IBM data/derived-asset distribution terms and retain applicable notices.
3. Create/commit/push only the approved source set; no such Git action occurred here.
4. Supply the exact trusted demo assets through an approved release/private mechanism.
5. Build an isolated compatible runtime with approval for dependency installation.
6. Validate model loading and synthetic inference without reevaluating the consumed test.
7. Configure provider runtime, interface, resource and upload constraints; retain
   model checks and readonly SQLite behavior.
8. Review access control, transport, privacy, logs and upload retention before exposure.
9. Deploy only after approving the concrete public destination and contents.
10. Smoke-test synthetic scenarios and saved evidence. Do not use test scores for tuning.

Cloud infrastructure does not make application security automatic.
[Provider shared-responsibility guidance](https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/trust-and-security).
No production security, fairness, monitoring or prospective churn-performance claim
is justified by this local assessment.
