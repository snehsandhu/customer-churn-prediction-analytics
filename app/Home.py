"""Local multipage dashboard entry point; all model/data services are read-only."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import streamlit as st
from churn_analytics.dashboard.components import style

st.set_page_config(page_title='Customer Churn | IBM Telco example',page_icon=':material/analytics:',layout='wide')
style()
with st.sidebar:
    st.markdown('### Customer churn')
    st.caption('Analytics & prediction portfolio')
    st.caption('Public IBM Telco example · Local demonstration')
page=st.navigation([
    st.Page('pages/1_Executive_Overview.py',title='Executive Overview',icon=':material/dashboard:',default=True),
    st.Page('pages/2_Customer_Churn_Analytics.py',title='Customer Analytics',icon=':material/query_stats:'),
    st.Page('pages/3_SQL_Business_Insights.py',title='SQL Business Insights',icon=':material/database:'),
    st.Page('pages/4_Customer_Churn_Prediction.py',title='Customer Prediction',icon=':material/person_search:'),
    st.Page('pages/5_Model_Performance.py',title='Model Performance',icon=':material/model_training:'),
    st.Page('pages/6_Reports_and_Downloads.py',title='Reports & Downloads',icon=':material/download:'),
])
page.run()
with st.sidebar:
    st.divider()
    st.caption('Frozen threshold: 50% · Test evaluation completed once. No model training in this application.')
