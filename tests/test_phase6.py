"""Native Streamlit AppTest smoke/interaction checks; no held-out scoring."""
from unittest.mock import Mock
import pytest
from streamlit.testing.v1 import AppTest
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from churn_analytics.data import PROJECT_ROOT


@pytest.fixture(autouse=True)
def no_training_or_evaluation(monkeypatch):
    for cls in [Pipeline,RandomForestClassifier]:monkeypatch.setattr(cls,'fit',Mock(side_effect=AssertionError('Dashboard must never fit')))
    monkeypatch.setattr('churn_analytics.final_evaluation.evaluate_final_once',Mock(side_effect=AssertionError('No held-out reevaluation')))
    monkeypatch.setattr('churn_analytics.phase5.run_phase5',Mock(side_effect=AssertionError('No model development')))


def start():
    app=AppTest.from_file(PROJECT_ROOT/'app/Home.py',default_timeout=40).run()
    assert not app.exception and not app.error
    return app


@pytest.mark.parametrize('page',['1_Executive_Overview','2_Customer_Churn_Analytics','3_SQL_Business_Insights',
    '4_Customer_Churn_Prediction','5_Model_Performance','6_Reports_and_Downloads'])
def test_each_page_imports_and_renders_without_errors(page):
    app=start().switch_page('pages/'+page+'.py').run()
    assert not app.exception and not app.error
    assert len(app.title)==1


def test_coordinated_filters_handle_no_customers():
    app=start().switch_page('pages/2_Customer_Churn_Analytics.py').run()
    app.multiselect(key='filter_Contract').set_value([]).run()
    assert not app.exception and not app.error
    assert any('No customers match' in text.value for text in app.info)
    assert app.metric[0].value=='0'


def test_prediction_form_submit_reset_and_service_dependencies():
    app=start().switch_page('pages/4_Customer_Churn_Prediction.py').run()
    next(b for b in app.button if b.label=='Predict churn').click().run()
    assert not app.exception and not app.error
    assert any(m.label=='Estimated churn probability' for m in app.metric)
    assert next(m.value for m in app.metric if m.label=='Frozen threshold')=='50%'
    app.selectbox(key='pred_InternetService').select('No').run()
    app.selectbox(key='pred_PhoneService').select('No').run()
    assert app.selectbox(key='pred_TechSupport').value=='No internet service'
    assert app.selectbox(key='pred_MultipleLines').value=='No phone service'
    next(b for b in app.button if b.label=='Predict churn').click().run()
    assert not app.exception and not app.error
    next(b for b in app.button if b.label=='Reset inputs').click().run()
    assert app.selectbox(key='pred_InternetService').value=='DSL'
    assert not any(m.label=='Estimated churn probability' for m in app.metric)


def test_sql_switches_to_window_question_without_write_controls():
    app=start().switch_page('pages/3_SQL_Business_Insights.py').run()
    app.selectbox[0].select('13_adjacent_tenure_bands').run()
    assert not app.exception and not app.error
    assert app.dataframe
    assert not app.text_area
