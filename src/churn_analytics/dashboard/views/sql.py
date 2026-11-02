import streamlit as st
from ..components import shell,rate_chart
from ..sql import SQL_QUESTIONS,SQL_EXPLANATIONS,query_business,query_text
from ..exports import aggregate_csv


def render():
    """Curated aggregate-only exploration over the original read-only database."""
    shell('SQL business insights','Explore business questions answered by the existing SQLite analytics. Only approved aggregate queries are available.')
    name=st.selectbox('Business question',list(SQL_QUESTIONS),format_func=SQL_QUESTIONS.get)
    minimum=100
    if name in {'11_service_combinations','14_substantial_above_average_segments'}:
        minimum=st.slider('Minimum customers per group',1,1000,100,step=1)
    st.write(SQL_EXPLANATIONS[name])
    with st.expander('View the SQL and how to read it'):
        st.code(query_text(name),language='sql')
        st.caption('Named parameters are bound separately. The connection is read-only; no free-form SQL or write commands are offered.')
    result=query_business(name,min_customers=minimum)
    if result.empty:
        st.info('No groups meet the selected minimum sample size.')
        return
    st.dataframe(result,width='stretch',hide_index=True)
    if 'churn_pct' in result and 'customer_count' in result:
        columns=[c for c in ['Contract','TenureGroup','PaymentMethod','InternetService','charge_quartile'] if c in result]
        if columns:
            graph=result.copy()
            graph['Segment']=graph[columns].astype(str).agg(' · '.join,axis=1)
            rate_chart(graph,'Segment','Observed churn in approved SQL groups',rate='churn_pct')
    elif name=='01_customer_kpis':
        st.caption('The summary reconciles with the Executive Overview. Charges are sample units, not lost revenue.')
    st.download_button('Download aggregate query results',aggregate_csv(result),name+'.csv','text/csv')
    st.caption('Counts and percentages describe the snapshot. Rankings and neighboring tenure bands do not establish causes or historical changes.')
