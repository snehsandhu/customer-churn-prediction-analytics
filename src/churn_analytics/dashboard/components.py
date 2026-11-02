"""Shared native Streamlit layout and interactive charts over aggregate data."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from .artifacts import DashboardError
from .data import FILTER_COLUMNS, segment_table, snapshot_kpis

TEAL = "#0f766e"
NAVY = "#183451"
LABELS = {"Contract":"Contract", "InternetService":"Internet service", "PaymentMethod":"Payment method",
 "TenureGroup":"Tenure band", "MonthlyChargeGroup":"Monthly charge band", "SeniorCitizen":"Senior citizen",
 "PaperlessBilling":"Paperless billing", "TechSupport":"Technical support", "OnlineSecurity":"Online security"}


def shell(title: str, subtitle: str):
    """Consistent page hierarchy with concise public-data context."""
    st.caption("CUSTOMER CHURN  /  PUBLIC IBM TELCO EXAMPLE")
    st.title(title)
    st.write(subtitle)


def style():
    """Small static layout enhancements; user content is never injected as HTML."""
    st.markdown('''<style>
    .block-container {max-width:1320px;padding-top:4.25rem;padding-bottom:3rem;}
    h1 {letter-spacing:-.035em;} h2,h3 {letter-spacing:-.02em;}
    [data-testid="stMetricValue"] {font-variant-numeric:tabular-nums;color:#183451;}
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
        font-size:clamp(1.15rem,2.5vw,2rem)!important;
        white-space:normal!important;overflow:visible!important;
    }
    [data-testid="stMetricLabel"] p {
        white-space:normal!important;overflow:visible!important;text-overflow:clip!important;
    }
    [data-testid="stSidebar"] {background:#eef3f7;}
    div[data-testid="stMetric"] {padding:.4rem .15rem;}
    </style>''', unsafe_allow_html=True)


def kpi_cards(frame, *, compact=False):
    """Display established snapshot KPI definitions, including undefined empty rates."""
    k = snapshot_kpis(frame)
    entries = [("Customers", f"{k['total_customers']:,}"), ("Churned", f"{k['churn_count']:,}"),
        ("Retained / no churn", f"{k['retained_count']:,}"), ("Observed churn", f"{k['churn_proportion']:.2%}" if len(frame) else "—"),
        ("Mean monthly charges", f"{k['mean_monthly_charges']:.2f}" if len(frame) else "—"),
        ("Mean tenure (months)", f"{k['mean_tenure_months']:.2f}" if len(frame) else "—")]
    for start in (0, 3):
        for col, (label, value) in zip(st.columns(3), entries[start:start+3]):
            with col.container(border=True): st.metric(label, value)
    st.caption("Charges use dataset currency units. Retained means no churn label in this snapshot, not measured long-term retention.")


def rate_chart(table, column, title, *, rate="churn_rate", count="customer_count"):
    """Within-segment proportions on a zero-based axis, with group sizes on hover."""
    values = table.loc[table[count].gt(0)].copy()
    values[column] = values[column].astype(str)
    if rate == "churn_pct": values["churn_rate"] = values[rate]/100
    chart = alt.Chart(values).mark_bar(color=TEAL, cornerRadiusEnd=3).encode(
        x=alt.X("churn_rate:Q", title="Observed churn within segment", scale=alt.Scale(domain=[0,1]), axis=alt.Axis(format="%")),
        y=alt.Y(f"{column}:N", title=None, sort=None, axis=alt.Axis(labelLimit=240)),
        tooltip=[alt.Tooltip(f"{column}:N",title="Segment"), alt.Tooltip(f"{count}:Q",title="Customers",format=","),
                 alt.Tooltip("churn_rate:Q",title="Churn rate",format=".2%")]).properties(height=max(180,len(values)*32),title=title)
    st.altair_chart(chart, width="stretch")
    st.caption("Denominator: all customers within each displayed segment. Hover for sample sizes; groups below 30 are exploratory.")


def distribution(frame):
    """Aggregate class counts and percentages; no individual customer data sent."""
    k = snapshot_kpis(frame)
    table = pd.DataFrame({"Outcome":["No churn","Churn"], "Customers":[k['retained_count'],k['churn_count']]})
    table['Share'] = table.Customers / max(1,len(frame))
    chart=alt.Chart(table).mark_bar(cornerRadiusEnd=4).encode(x=alt.X('Customers:Q',title='Customers',scale=alt.Scale(zero=True)),
        y=alt.Y('Outcome:N',title=None),color=alt.Color('Outcome:N',scale=alt.Scale(domain=['No churn','Churn'],range=[NAVY,TEAL]),legend=None),
        tooltip=['Outcome','Customers',alt.Tooltip('Share:Q',format='.2%')]).properties(height=180,title='Snapshot churn distribution')
    st.altair_chart(chart,width='stretch')


def filters(frame):
    """All selected means unrestricted; no selection deliberately yields no rows."""
    selected={}
    with st.sidebar:
        st.subheader("Explore the snapshot")
        st.caption("Filters apply together on this page. Clear a selection to return no matching customers.")
        for column in FILTER_COLUMNS:
            values=frame[column].dropna().unique().tolist()
            selected[column]=st.multiselect(LABELS[column],values,default=values,key='filter_'+column,
                format_func=(lambda v: 'Yes' if v==1 else 'No') if column=='SeniorCitizen' else str)
    return selected


def run_page(function):
    """Keep operational errors actionable without stack traces or filesystem paths."""
    try: function()
    except DashboardError as exc: st.error(str(exc))
    except Exception: st.error("This page is temporarily unavailable. Check the supplied local application files and compatible environment.")
