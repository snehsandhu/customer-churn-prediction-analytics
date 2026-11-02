import altair as alt
import pandas as pd
import streamlit as st
from ..cache import analytics_frame
from ..components import shell, kpi_cards, filters, distribution, rate_chart, LABELS, TEAL
from ..data import filter_customers, segment_table, numerical_bins
from ..exports import aggregate_csv


def render():
    """Coordinated filters affect every count, distribution and segment denominator."""
    shell('Customer churn analytics','Choose customer groups in the sidebar. Every view below reflects the same filtered population.')
    full=analytics_frame();frame=filter_customers(full,filters(full))
    kpi_cards(frame)
    if frame.empty:
        st.info('No customers match the selected filters. Restore categories in the sidebar to continue.')
        return
    column=st.selectbox('Compare observed churn by',list(LABELS),format_func=LABELS.get)
    table=segment_table(frame,column)
    left,right=st.columns(2)
    with left: rate_chart(table,column,'Within-segment churn in the filtered snapshot')
    with right: distribution(frame)
    shown=table.copy()
    st.dataframe(shown,width='stretch',hide_index=True)
    if shown.small_segment.any(): st.warning('Some groups have fewer than 30 customers. Their rates are unstable; inspect counts before drawing conclusions.')
    st.download_button('Download filtered segment summary',aggregate_csv(table),'filtered_segments.csv','text/csv')
    st.subheader('Numerical distributions')
    variable=st.selectbox('Measure',['tenure','MonthlyCharges','TotalCharges'],format_func=lambda v:{'tenure':'Tenure (months)','MonthlyCharges':'Monthly charges (dataset units)','TotalCharges':'Total charges (dataset units)'}[v])
    missing=int(frame[variable].isna().sum())
    st.caption(f'{len(frame)-missing:,} observed values; {missing:,} missing. Missing TotalCharges remains missing and is excluded from this distribution.')
    # Aggregate in Python before sending a histogram to the browser: no raw rows.
    grouped=numerical_bins(frame,variable)
    if not grouped.empty:
        chart=alt.Chart(grouped).mark_bar().encode(x=alt.X('Band:N',sort=None,title=variable+' bands',axis=alt.Axis(labelAngle=-40)),
            y=alt.Y('Customers:Q',title='Customers'),color=alt.Color('Outcome:N',scale=alt.Scale(range=['#183451',TEAL])),tooltip=['Band','Outcome','Customers']).properties(height=280,title='Customer counts by numerical band and churn label')
        st.altair_chart(chart,width='stretch')
    st.subheader('Contract and tenure together')
    joint=segment_table(frame,['Contract','TenureGroup']).query('customer_count > 0').copy()
    for c in ['Contract','TenureGroup']:joint[c]=joint[c].astype(str)
    heat=alt.Chart(joint).mark_rect().encode(x=alt.X('TenureGroup:N',title='Tenure band'),y=alt.Y('Contract:N',title='Contract'),
        color=alt.Color('churn_rate:Q',title='Churn proportion',scale=alt.Scale(domain=[0,1],scheme='tealblues')),
        tooltip=['Contract','TenureGroup',alt.Tooltip('customer_count:Q',title='Customers'),alt.Tooltip('churn_rate:Q',format='.2%')]).properties(height=200)
    st.altair_chart(heat,width='stretch')
    st.caption('Each heatmap cell uses its own customer count as denominator. Empty cells are omitted. Differences are descriptive, not causal.')
