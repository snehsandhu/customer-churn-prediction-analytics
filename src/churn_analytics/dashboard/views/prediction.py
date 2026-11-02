import pandas as pd
import streamlit as st
from ...cleaning import INTERNET_ADDONS
from ...modeling import FEATURES
from ..cache import predictor_resource
from ..components import shell
from ..prediction import DEFAULT_RECORD,score_inputs,parse_upload,template_csv
from ..artifacts import DashboardError


def clear_result():
    st.session_state.pop('single_result',None)


def reset():
    """Reset only this session's widgets/results; no files or shared cache touched."""
    for key,value in DEFAULT_RECORD.items():st.session_state['pred_'+key]=value
    clear_result()
    st.session_state.pop('batch_result',None)
    st.session_state.pop('batch_upload',None)


def render():
    """Proper submit-driven scoring, with service dependencies resolved in controls."""
    shell('Customer churn prediction','Try an illustrative customer scenario with the frozen Random Forest. No entered data is added to training or saved to disk.')
    st.caption('This model includes demographic fields from the public dataset; suitability and fairness have not been established. The result is not a causal explanation or an individual guarantee.')
    st.button('Reset inputs',on_click=reset)
    for column,value in DEFAULT_RECORD.items():st.session_state.setdefault('pred_'+column,value)
    left,right=st.columns(2)
    with left: phone=st.selectbox('Phone service',['Yes','No'],key='pred_PhoneService',on_change=clear_result)
    with right: internet=st.selectbox('Internet service',['DSL','Fiber optic','No'],key='pred_InternetService',on_change=clear_result)
    st.caption('Choose services first. Dependent fields use the trained “No phone service” / “No internet service” representation when not applicable. Prediction runs only after submission.')
    record={'PhoneService':phone,'InternetService':internet}
    with st.form('customer_prediction'):
        a,b,c=st.columns(3)
        with a:
            record['tenure']=st.number_input('Tenure (months)',0,72,key='pred_tenure',help='Whole months; app range follows the 0–72-month snapshot.')
            record['MonthlyCharges']=st.number_input('Monthly charges (dataset units)',0.0,200.0,step=.05,key='pred_MonthlyCharges',help='App accepts 0–200. Observed sample charges are approximately 18–119; outside-sample scenarios have additional uncertainty.')
            record['Contract']=st.selectbox('Contract',['Month-to-month','One year','Two year'],key='pred_Contract')
            record['PaymentMethod']=st.selectbox('Payment method',['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'],key='pred_PaymentMethod')
            record['PaperlessBilling']=st.selectbox('Paperless billing',['Yes','No'],key='pred_PaperlessBilling')
        with b:
            record['gender']=st.selectbox('Gender',['Female','Male'],key='pred_gender')
            record['SeniorCitizen']=st.selectbox('Senior citizen',[0,1],format_func=lambda x:'Yes' if x else 'No',key='pred_SeniorCitizen')
            record['Partner']=st.selectbox('Partner',['No','Yes'],key='pred_Partner')
            record['Dependents']=st.selectbox('Dependents',['No','Yes'],key='pred_Dependents')
            options=['No','Yes'] if phone=='Yes' else ['No phone service']
            key='pred_MultipleLines'
            if st.session_state.get(key) not in options:st.session_state[key]=options[0]
            record['MultipleLines']=st.selectbox('Multiple phone lines',options,key=key,disabled=phone=='No')
        with c:
            for column in INTERNET_ADDONS:
                options=['No','Yes'] if internet!='No' else ['No internet service']
                key='pred_'+column
                if st.session_state.get(key) not in options:st.session_state[key]=options[0]
                labels={'OnlineSecurity':'Online security','OnlineBackup':'Online backup','DeviceProtection':'Device protection','TechSupport':'Technical support','StreamingTV':'Streaming TV','StreamingMovies':'Streaming movies'}
                record[column]=st.selectbox(labels[column],options,key=key,disabled=internet=='No')
        submitted=st.form_submit_button('Predict churn',type='primary')
    if submitted:
        try:st.session_state['single_result']=score_inputs(pd.DataFrame([record],columns=FEATURES),predictor_resource()).iloc[0].to_dict()
        except DashboardError as exc:
            clear_result();st.error(str(exc))
    if 'single_result' in st.session_state:
        result=st.session_state['single_result']
        with st.container(border=True):
            st.subheader('Most recent submitted scenario')
            a,b,c=st.columns(3)
            a.metric('Predicted classification',result['predicted_label'])
            b.metric('Estimated churn probability',f"{result['churn_probability']:.1%}")
            c.metric('Frozen threshold','50%')
            st.progress(float(result['churn_probability']))
            st.write('Probabilities at or above 50% receive the Churn label. This is a predictive estimate based on the public snapshot, not a guarantee or a measured effect of a retention action.')
            st.caption('Submit again after changing inputs. Global feature importance is available on Model Performance; it does not explain this individual scenario.')
    with st.expander('Batch prediction · feature-only CSV'):
        st.write('Up to 500 records and 1 MiB. Exactly 18 unique feature headers; no customer IDs, targets or historical test-file demos. Uploads stay in memory and are not globally cached.')
        st.download_button('Download illustrative input template',template_csv(),'illustrative_features.csv','text/csv')
        upload=st.file_uploader('Feature-only customer CSV',type=['csv'],key='batch_upload',on_change=lambda:st.session_state.pop('batch_result',None))
        if st.button('Predict uploaded records'):
            st.session_state.pop('batch_result',None)
            if upload is None:st.info('Choose a feature-only CSV first.')
            else:
                try:st.session_state['batch_result']=score_inputs(parse_upload(upload.getvalue()),predictor_resource())
                except DashboardError as exc:st.error(str(exc))
        if 'batch_result' in st.session_state:
            result=st.session_state['batch_result']
            st.dataframe(result,width='stretch',hide_index=True)
            st.download_button('Download prediction results',result.to_csv(index=False).encode(),'uploaded_predictions.csv','text/csv')
