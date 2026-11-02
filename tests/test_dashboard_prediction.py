"""Phase 6 inference uses invented records, never historical test rows."""
from io import BytesIO
import hashlib
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from churn_analytics.dashboard.artifacts import DashboardError,MODEL_PATH
from churn_analytics.dashboard.prediction import DEFAULT_RECORD,load_predictor,parse_upload,score_inputs,template_csv,validate_inputs,MAX_UPLOAD_BYTES
from churn_analytics.modeling import FEATURES
from churn_analytics.data import PROJECT_ROOT
from churn_analytics.phase5_data import sha256


class FakePredictor:
    threshold=.50
    def predict_proba(self,frame):
        probability=np.where(frame.SeniorCitizen.eq('1'),.75,.25)
        return np.column_stack([1-probability,probability])


def test_batch_order_numeric_string_normalization_and_frozen_threshold():
    frame=pd.DataFrame([DEFAULT_RECORD,{**DEFAULT_RECORD,'SeniorCitizen':1}],columns=FEATURES)
    original=frame.copy(deep=True)
    result=score_inputs(frame,FakePredictor())
    assert result.predicted_label.tolist()==['No Churn','Churn']
    assert result.row_index.tolist()==[1,2]
    frame.SeniorCitizen=frame.SeniorCitizen.astype(str)
    pd.testing.assert_frame_equal(result,score_inputs(frame,FakePredictor()))
    actual=parse_upload(original.to_csv(index=False).encode())
    pd.testing.assert_frame_equal(result,score_inputs(actual,FakePredictor()))
    changed=FakePredictor();changed.threshold=.3
    with pytest.raises(DashboardError):score_inputs(frame,changed)


@pytest.mark.parametrize('column,value', [('SeniorCitizen',2),('SeniorCitizen','1.0'),('SeniorCitizen',True),
    ('tenure',-1),('tenure',73),('tenure',1.5),('tenure',True),('MonthlyCharges',201),('MonthlyCharges',np.inf),
    ('Contract','month-to-month'),('InternetService','Fiber Optic'),('gender',None),('PhoneService','No')])
def test_invalid_inputs_fail_before_scoring(column,value):
    frame=pd.DataFrame([DEFAULT_RECORD],columns=FEATURES).astype(object)
    frame.loc[0,column]=value
    predictor=Mock(threshold=.5)
    with pytest.raises(DashboardError):score_inputs(frame,predictor)
    predictor.predict_proba.assert_not_called()


def test_exact_schema_and_inconsistent_internet_service():
    frame=pd.DataFrame([DEFAULT_RECORD],columns=FEATURES)
    invalid=[frame.drop(columns='tenure'),frame.assign(ChurnFlag=1),pd.concat([frame,frame[['tenure']]],axis=1),frame.iloc[:0]]
    for f in invalid:
        with pytest.raises(DashboardError):validate_inputs(f)
    frame.InternetService='No'
    with pytest.raises(DashboardError):validate_inputs(frame)


@pytest.mark.parametrize('content',[b'',b'wrong,header\n1,2\n',b'\xffinvalid',b'"unterminated',b'x'*(MAX_UPLOAD_BYTES+1)])
def test_malformed_uploads_are_rejected(content):
    with pytest.raises(DashboardError):parse_upload(content)


def test_upload_duplicate_headers_targets_row_limit_and_good_template():
    frame=pd.DataFrame([DEFAULT_RECORD],columns=FEATURES)
    raw=frame.to_csv(index=False)
    duplicate=raw.replace('tenure,MonthlyCharges','tenure,tenure',1)
    for value in [duplicate,frame.assign(Churn='Yes').to_csv(index=False),
                  pd.concat([frame]*501).to_csv(index=False),raw.splitlines()[0]+'\n']:
        with pytest.raises(DashboardError):parse_upload(value.encode())
    assert len(parse_upload(template_csv()))==1


def test_trusted_real_model_on_illustrative_records_never_fits_or_changes_artifacts(monkeypatch):
    """Integration: only invented DEFAULT_RECORD is scored by the trusted artifact."""
    for cls in [Pipeline,RandomForestClassifier]:monkeypatch.setattr(cls,'fit',Mock(side_effect=AssertionError('No training')))
    protected=[PROJECT_ROOT/MODEL_PATH,PROJECT_ROOT/'reports/phase5/test_consumption.json',PROJECT_ROOT/'data/processed/telco_churn_cleaned.csv']
    before={str(p):sha256(p) for p in protected}
    predictor=load_predictor()
    frame=pd.DataFrame([DEFAULT_RECORD],columns=FEATURES)
    first=score_inputs(frame,predictor)
    frame.SeniorCitizen=frame.SeniorCitizen.astype(str)
    pd.testing.assert_frame_equal(first,score_inputs(frame,predictor))
    assert first.churn_probability.between(0,1).all()
    assert before=={str(p):sha256(p) for p in protected}
