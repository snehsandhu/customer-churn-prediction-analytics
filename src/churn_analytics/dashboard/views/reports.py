import pandas as pd
import streamlit as st
from ..cache import analytics_frame
from ..components import shell
from ..artifacts import load_table,saved_performance
from ..data import snapshot_kpis,segment_table,SEGMENTS
from ..exports import aggregate_csv,public_summary
from ..sql import SQL_QUESTIONS,query_business


def render():
    """Curated public summaries and aggregate downloads, without raw-record access."""
    shell('Reports & downloads','Take away aggregate business summaries and saved model evidence. Customer identifiers, raw records and individual test predictions are excluded.')
    frame=analytics_frame();performance=saved_performance()
    with st.container(border=True):
        st.subheader('Executive summary')
        k=snapshot_kpis(frame)
        st.write(f"The snapshot contains {k['total_customers']:,} customers with {k['churn_proportion']:.2%} observed churn. The completed final evaluation belongs to the frozen tuned Random Forest at a 50% threshold.")
        st.download_button('Download public summary · JSON',public_summary(frame,performance),'churn_public_summary.json','application/json')
        st.download_button('Download snapshot KPI summary · CSV',aggregate_csv(pd.DataFrame([k])),'snapshot_kpis.csv','text/csv')
    a,b=st.columns(2)
    with a:
        st.subheader('Customer segments')
        segment=st.selectbox('Segment report',SEGMENTS)
        table=segment_table(frame,segment)
        st.dataframe(table,width='stretch',hide_index=True)
        st.download_button('Download segment report',aggregate_csv(table),'segment_summary.csv','text/csv')
    with b:
        st.subheader('Saved validation comparison')
        comparison=load_table('comparison')
        st.dataframe(comparison[['model_name','average_precision','roc_auc','f1']],width='stretch',hide_index=True)
        st.download_button('Download model comparison',aggregate_csv(comparison),'saved_validation_comparison.csv','text/csv')
    st.subheader('SQL business report')
    name=st.selectbox('Approved aggregate report',list(SQL_QUESTIONS),format_func=SQL_QUESTIONS.get)
    sql=query_business(name)
    st.dataframe(sql,width='stretch',hide_index=True)
    st.download_button('Download SQL aggregate report',aggregate_csv(sql),'sql_aggregate.csv','text/csv')
    st.subheader('Methodology & limitations')
    st.markdown('''The public IBM Telco example is a single customer snapshot. Phase 2 cleaning preserves all records and retains missing TotalCharges rather than assuming zero. SQL and Pandas share definitions and segment denominators. The dashboard reads the existing analytical data and SQLite database without updating them.

Model selection used original training/validation memberships and the same 18 eligible features. The selected model and 50% threshold were frozen before one final test evaluation. That test is consumed. This application displays saved aggregate evidence and scores user-provided scenarios only; it never retrains or reevaluates historical test records.

Earlier full-data EDA, uncertain prediction-time feature availability, demographic suitability, sparse segments, probability calibration and dataset shift limit interpretation. Global importance is not an individual causal explanation. No historical trends, retained-customer gains, revenue savings or retention ROI are established.''')
