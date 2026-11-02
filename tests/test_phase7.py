"""Narrow Phase 7 regressions: ordered charts, upload order and portable navigation."""
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from churn_analytics.data import PROJECT_ROOT
from churn_analytics.dashboard.data import numerical_bins
from churn_analytics.dashboard.prediction import DEFAULT_RECORD,parse_upload,score_inputs
from churn_analytics.modeling import FEATURES


def test_histogram_preserves_numeric_order_counts_and_missing_coverage():
    frame=pd.DataFrame({'MonthlyCharges':[20.,29.,65.,108.,118.,np.nan],'ChurnFlag':[0,1,0,1,0,1]})
    original=frame.copy(deep=True)
    result=numerical_bins(frame,'MonthlyCharges')
    lower=[float(label[1:].split(',')[0]) for label in result.Band]
    assert lower==sorted(lower)
    assert result.Customers.sum()==5
    assert result.groupby('Outcome').Customers.sum().to_dict()=={'Churn':2,'No churn':3}
    pd.testing.assert_frame_equal(frame,original)
    assert numerical_bins(frame.iloc[:0],'MonthlyCharges').empty


def test_reordered_csv_columns_preserve_original_record_order():
    class Predictor:
        threshold=.5
        def predict_proba(self,frame):
            p=np.where(frame.SeniorCitizen.eq('1'),.8,.2)
            return np.column_stack([1-p,p])
    frame=pd.DataFrame([DEFAULT_RECORD,{**DEFAULT_RECORD,'SeniorCitizen':1}],columns=FEATURES)
    expected=score_inputs(frame,Predictor())
    actual=score_inputs(parse_upload(frame.iloc[:,::-1].to_csv(index=False).encode()),Predictor())
    pd.testing.assert_frame_equal(actual,expected)
    assert actual.row_index.tolist()==[1,2]


def test_app_entry_and_registered_page_work_outside_project_cwd(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    app=AppTest.from_file(PROJECT_ROOT/'app/Home.py',default_timeout=40).run()
    assert not app.error and not app.exception
    app.switch_page('pages/4_Customer_Churn_Prediction.py').run()
    assert not app.error and not app.exception
    assert any(button.label=='Predict churn' for button in app.button)


def test_kpi_units_live_in_labels_instead_of_overflowing_values():
    app=AppTest.from_file(PROJECT_ROOT/'app/Home.py',default_timeout=40).run()
    assert not app.error and not app.exception
    tenure=next(metric for metric in app.metric if metric.label=='Mean tenure (months)')
    assert tenure.value=='32.37'
