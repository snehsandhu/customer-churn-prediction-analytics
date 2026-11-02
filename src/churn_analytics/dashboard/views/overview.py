import streamlit as st
from ..cache import analytics_frame
from ..components import shell, kpi_cards, distribution, rate_chart
from ..data import insights, segment_table


def render():
    """Business snapshot with established KPIs and denominator-aware comparisons."""
    shell("Understand customer churn", "Explore the public IBM Telco snapshot, compare observed segments, and try the frozen prediction model.")
    st.info("A customer snapshot—not a historical trend. Observed associations suggest questions to investigate; they do not prove retention effects.")
    frame=analytics_frame()
    kpi_cards(frame)
    st.subheader("Where churn is concentrated")
    left,right=st.columns(2)
    with left: distribution(frame)
    with right: rate_chart(segment_table(frame,'Contract'),'Contract','Churn by contract')
    for columns in [('TenureGroup','MonthlyChargeGroup'),('PaymentMethod','InternetService')]:
        for panel,column in zip(st.columns(2),columns):
            with panel: rate_chart(segment_table(frame,column),column,'Churn by '+{'TenureGroup':'tenure','MonthlyChargeGroup':'monthly charges','PaymentMethod':'payment method','InternetService':'internet service'}[column])
    st.subheader("Business signals to investigate")
    for text in insights(frame): st.write('• '+text)
    st.caption("Potential next step: test contract, onboarding or service-support hypotheses prospectively with measured costs and treatment response. This dataset does not establish savings or ROI.")
