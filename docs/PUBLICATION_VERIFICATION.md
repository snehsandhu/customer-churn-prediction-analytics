# Publication preparation verification

Date: 2026-11-02 (Asia/Kolkata). **Source candidate prepared; Gate 1 pending.**
No repository creation, Git initialization, commit, push, public visibility change,
package installation, model retraining or held-out reevaluation was performed.

## Fresh execution

| Check | Actual result |
|---|---|
| Complete unmodified suite, isolated owner-held project copy | **230 passed, 0 failed, 0 skipped**, 24.55 s |
| Asset-independent subset, exact public staging source | **100 passed, 0 failed, 0 skipped**, 10.01 s |
| Private overlay using staged code + original trusted assets | All six AppTest pages; nine SQL questions; rejected SQL write; invented-customer form passed |
| Model protection in overlay smoke | Pipeline/RF/LR fitting blocked; final evaluation blocked; zero guarded calls |
| Browser | All six pages displayed/captured, plus real form submission/result using invented inputs |
| Notebook publication privacy | Four source copies inspected; outputs/counts/attachments cleared; no notebook executed |

Full suite: `pytest -q -p no:cacheprovider --basetemp <isolated-temporary-directory>`
with automatic plugin loading disabled. The private runner used the existing
Python environment through an explicitly selected site-packages path and bypassed
editable `.pth` processing, so imports resolved to the isolated copy. Exact private
commands/logs are retained by the owner rather than published with machine paths.
Existing synthetic unit fixtures fit temporary test estimators; no saved IBM model
was retrained, and no consumed IBM test rows were rescored.

Executed asset-independent subset (equivalent pytest selection from release root):

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider \
  tests/test_data.py tests/test_cleaning.py tests/test_analytics.py \
  tests/test_database.py tests/test_sql_analytics.py tests/test_canonical_acquisition.py
```

The ordinary full-suite command remains `python -m pytest -q`, but application/
frozen-bundle integration checks need the separately supplied trusted assets.
Do not infer 230 passing tests or a complete dashboard from an assets-free clone.

## Environment and limitations

Existing macOS Python 3.13.9 environment: NumPy 2.5.4, Pandas 2.3.3, scikit-learn
1.9.1, joblib 1.6.0, Streamlit 1.66.1, Altair 6.3.0. Imports were executed.
Dependency files are unchanged; no fresh installation or cross-platform build was
performed. Exact sklearn compatibility is enforced by the existing trusted bundle.
No public hosting, GitHub Pages or Streamlit deployment occurred.

The proposed text contents were scanned for known tokens, private keys, credential
literals/URLs, personal machine paths, emails and IBM customer IDs. No matches
were found in the initial completed content scan; a final frozen file manifest is
retained privately before Gate 1. This best-effort review cannot prove absence of
all secrets or establish authorship/legal rights. Public Markdown file links are
checked against the candidate tree; notebook paths to withheld historical artifacts
are intentional and disclosed, not promises of executable asset-free notebooks.

All runtime source, tests and SQL implementations remain byte-identical to the
original. Publication-only README/docs, privacy-cleared notebook copies, this
summary and screenshots are documented derivatives. The original complete project
and historical verification files remain local and unchanged. See
[licensing](PUBLIC_RELEASE_LICENSING.md) and [asset policy](PUBLIC_RELEASE_ASSETS.md).
