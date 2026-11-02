"""Phase 6 data/SQL/report checks, using small synthetic snapshots where possible."""
from contextlib import closing
import json
import shutil
import sqlite3

import pandas as pd
import pytest

from churn_analytics.analytics import customer_kpis
from churn_analytics.cleaning import save_cleaned_dataset
from churn_analytics.database import load_snapshot
from churn_analytics.dashboard.artifacts import DashboardError,load_table,figure_bytes,saved_performance,verify_final_bundle,MODEL_PATH
from churn_analytics.dashboard.data import filter_customers,load_analytics,segment_table,snapshot_kpis
from churn_analytics.dashboard.exports import aggregate_csv,public_summary
from churn_analytics.dashboard.sql import SQL_QUESTIONS,open_readonly,query_business,query_text
from churn_analytics.data import PROJECT_ROOT
from churn_analytics.phase5_data import sha256


@pytest.fixture
def analytical_root(tmp_path,cleaned_customers):
    root=tmp_path/'analytics';source=root/'data/processed/telco_churn_cleaned.csv'
    save_cleaned_dataset(cleaned_customers,source)
    (root/'reports/phase4').mkdir(parents=True)
    (root/'reports/phase4/split_manifest.json').write_text(json.dumps({'dataset_sha256':sha256(source)}))
    load_snapshot(cleaned_customers,root/'data/database/customer_churn.db')
    return root


@pytest.fixture
def bundle_root(tmp_path):
    """Copy trusted existing evidence only; no fitting or historical row scoring."""
    root=tmp_path/'bundle'
    for rel in [MODEL_PATH,*[f'reports/phase5/{name}.json' for name in
        ['final_selection_manifest','final_model_metadata','test_consumption','final_test_metrics']]]:
        target=root/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PROJECT_ROOT/rel,target)
    return root


def test_existing_kpis_filter_denominators_and_no_identifiers(analytical_root,cleaned_customers):
    before=sha256(analytical_root/'data/processed/telco_churn_cleaned.csv')
    frame=load_analytics(analytical_root)
    assert 'customerID' not in frame
    assert snapshot_kpis(frame)==customer_kpis(cleaned_customers)
    selected=filter_customers(frame,{'Contract':['Month-to-month'],'InternetService':['DSL','Fiber optic']})
    expected=frame.loc[frame.Contract.eq('Month-to-month') & frame.InternetService.isin(['DSL','Fiber optic'])]
    pd.testing.assert_frame_equal(selected,expected)
    table=segment_table(frame,'Contract').set_index('Contract')
    assert table.loc['Month-to-month','churn_rate']==pytest.approx(1/3)
    assert snapshot_kpis(filter_customers(frame,{'Contract':[]}))['total_customers']==0
    with pytest.raises(DashboardError):filter_customers(frame,{'customerID':['C0']})
    with pytest.raises(DashboardError):filter_customers(frame,{'Contract':['TYPO']})
    assert before==sha256(analytical_root/'data/processed/telco_churn_cleaned.csv')


def test_missing_or_altered_snapshot_fails_without_regeneration(analytical_root):
    path=analytical_root/'data/processed/telco_churn_cleaned.csv'
    path.write_text('corrupted')
    with pytest.raises(DashboardError):load_analytics(analytical_root)
    assert path.read_text()=='corrupted'


def test_curated_sql_reconciles_and_stays_readonly(analytical_root,cleaned_customers):
    path=analytical_root/'data/database/customer_churn.db';before=sha256(path)
    result=query_business('01_customer_kpis',root=analytical_root).iloc[0]
    k=customer_kpis(cleaned_customers)
    assert result.total_customers==k['total_customers']
    assert result.churn_pct==pytest.approx(100*k['churn_proportion'])
    for name in SQL_QUESTIONS:
        table=query_business(name,min_customers=1,root=analytical_root)
        assert 'customerID' not in table
        assert aggregate_csv(table)
        assert 'SELECT' in query_text(name)
    for name in ['16_customer_charge_positions','../../secrets','DROP TABLE customer_snapshot','unknown']:
        with pytest.raises(DashboardError):query_business(name,root=analytical_root)
        with pytest.raises(DashboardError):query_text(name)
    with closing(open_readonly(analytical_root)) as conn:
        for sql in ['DELETE FROM customer_snapshot','UPDATE customer_snapshot SET tenure=1',
                    'DROP TABLE customer_snapshot',"ATTACH DATABASE ':memory:' AS extra",'PRAGMA query_only=OFF']:
            with pytest.raises(sqlite3.DatabaseError):conn.execute(sql)
    assert sha256(path)==before


def test_missing_database_and_invalid_parameters(analytical_root):
    with pytest.raises(DashboardError):query_business('01_customer_kpis',root=analytical_root/'missing')
    with pytest.raises(DashboardError):query_business('11_service_combinations',root=analytical_root,min_customers=-1)


@pytest.mark.parametrize('problem',['missing_model','model_hash','metadata','threshold','ledger','path'])
def test_artifact_failures_are_safe_and_no_paths_leak(bundle_root,problem):
    root=bundle_root
    if problem=='missing_model':(root/MODEL_PATH).unlink()
    elif problem=='model_hash':(root/MODEL_PATH).write_bytes(b'corrupted model')
    else:
        name='test_consumption' if problem=='ledger' else 'final_model_metadata'
        path=root/f'reports/phase5/{name}.json'
        value=json.loads(path.read_text())
        if problem=='metadata':path.write_text('{broken');value=None
        elif problem=='threshold':value['threshold']=.3
        elif problem=='path':value['artifact_path']='../../private.joblib'
        elif problem=='ledger':value['test_consumed']=False
        if value is not None:path.write_text(json.dumps(value))
    with pytest.raises(DashboardError) as exc:verify_final_bundle(root)
    assert str(root) not in str(exc.value)


def test_saved_aggregate_evidence_is_scrubbed_and_report_allowlists(bundle_root,cleaned_customers):
    summary=saved_performance(bundle_root)
    assert summary['test_metrics']['average_precision']==pytest.approx(.6631535733700588)
    assert summary['confusion_matrix']==[[925,110],[185,189]]
    public=json.loads(public_summary(cleaned_customers,summary))
    assert 'training_ids' not in json.dumps(public) and 'customerID' not in json.dumps(public)
    assert summary['threshold']==.5 and summary['test_consumed']
    for name in ['../data/raw','final_test_predictions','../../']:
        with pytest.raises(DashboardError):load_table(name,bundle_root)
        with pytest.raises(DashboardError):figure_bytes(name,bundle_root)
    with pytest.raises(DashboardError):load_table('comparison',bundle_root)
    with pytest.raises(DashboardError):figure_bytes('validation_roc',bundle_root)


def test_aggregate_export_blocks_records_and_escapes_spreadsheet_formulas():
    with pytest.raises(DashboardError):aggregate_csv(pd.DataFrame({'customerID':['X']}))
    with pytest.raises(DashboardError):aggregate_csv(pd.DataFrame({'ChurnFlag':[1]}))
    content=aggregate_csv(pd.DataFrame({'segment':['=1+1','ordinary'],'customer_count':[2,3]})).decode()
    assert "'=1+1" in content and 'ordinary' in content
