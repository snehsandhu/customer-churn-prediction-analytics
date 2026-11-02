import altair as alt
import pandas as pd
import streamlit as st
from ..artifacts import load_table,saved_performance,figure_bytes,DashboardError,METRIC_KEYS
from ..components import shell,TEAL,NAVY


def render():
    """Present stored aggregate evidence only; never read per-customer predictions."""
    shell('Model performance','Review the saved validation comparison and the completed one-time final evaluation. Nothing on this page evaluates or selects a model.')
    evidence=saved_performance()
    st.success('Frozen final model: tuned Random Forest · Threshold: 50% · Final test: consumed')
    validation,final,interpretation,curves=st.tabs(['Validation comparison','Final test summary','Global importance','Saved curves'])
    with validation:
        table=load_table('comparison')
        st.caption('Same 1,409 validation customers and 18 features for every model. Highest unrounded validation Average Precision selected the final candidate before test access.')
        metric=st.selectbox('Validation metric',list(METRIC_KEYS[:6]),format_func=lambda x:x.replace('_',' ').title())
        chart=alt.Chart(table).mark_bar(color=TEAL).encode(x=alt.X(f'{metric}:Q',scale=alt.Scale(domain=[0,1]),title=metric.replace('_',' ').title()),
            y=alt.Y('model_name:N',title=None),tooltip=['model_name',alt.Tooltip(f'{metric}:Q',format='.6f')]).properties(height=220)
        st.altair_chart(chart,width='stretch')
        st.dataframe(table[['model_name',*METRIC_KEYS[:6]]],width='stretch',hide_index=True)
        st.caption('Small validation differences do not prove general superiority. Ranking AP/ROC-AUC differs from threshold-dependent precision, recall and F1.')
    with final:
        st.subheader('Saved final test results · 1,409 customers')
        for keys in [('accuracy','precision','recall'),('f1','roc_auc','average_precision')]:
            for col,key in zip(st.columns(3),keys):col.metric(key.replace('_',' ').title(),f"{evidence['test_metrics'][key]:.6f}")
        st.caption(f"Brier score: {evidence['test_metrics']['brier_score']:.6f} · Log loss: {evidence['test_metrics']['log_loss']:.6f}. Lower probability loss is better on the same sample; it is not a guarantee of future calibration.")
        matrix=evidence['confusion_matrix']
        chart_data=pd.DataFrame([{'Actual':a,'Predicted':p,'Customers':matrix[i][j]} for i,a in enumerate(['No churn','Churn']) for j,p in enumerate(['No churn','Churn'])])
        base=alt.Chart(chart_data).encode(x=alt.X('Predicted:N',title='Predicted label'),y=alt.Y('Actual:N',title='Actual label'))
        heat=base.mark_rect().encode(color=alt.Color('Customers:Q',scale=alt.Scale(scheme='tealblues')),tooltip=['Actual','Predicted','Customers'])
        text=base.mark_text(fontSize=22).encode(text='Customers:Q',
            color=alt.condition(alt.datum.Customers > chart_data.Customers.max()/2,
                                alt.value('white'),alt.value(NAVY)))
        st.altair_chart((heat+text).properties(height=240,title='Saved confusion matrix'),width='stretch')
        tn,fp=matrix[0];fn,tp=matrix[1]
        st.write(f'**True negatives ({tn})**: no-churn customers labeled no churn. **False positives ({fp})**: no-churn customers labeled churn. **False negatives ({fn})**: churned customers missed. **True positives ({tp})**: churned customers identified.')
        st.info('Accuracy alone can hide missed churners. Outreach costs, treatment response and capacity are needed to evaluate business value; missed churn labels are not proven preventable losses.')
    with interpretation:
        st.subheader('Global model reliance, not a personal explanation')
        table=load_table('permutation').sort_values('ap_decrease_mean',ascending=False)
        st.dataframe(table[['feature','ap_decrease_mean','ap_decrease_std']],hide_index=True,width='stretch')
        st.write('Permutation importance measures validation AP decrease after shuffling an original input column. Five fixed-seed repeats show shuffle variability, not a population confidence interval. Correlated substitutes and nonphysical service combinations limit interpretation.')
        for name in ['rf_permutation_importance','rf_impurity_importance']:
            try:st.image(figure_bytes(name))
            except DashboardError as exc:st.info(str(exc))
        st.caption('Impurity importance can favor features with many split opportunities. These global statistics do not establish causality and do not explain a particular customer’s predicted probability.')
    with curves:
        for name,title in [('validation_roc','ROC ranking'),('validation_precision_recall','Precision–recall ranking'),('validation_calibration','Probability reliability'),('validation_threshold_tradeoffs','Threshold trade-offs')]:
            st.subheader(title)
            try:st.image(figure_bytes(name))
            except DashboardError as exc:st.info(str(exc))
        st.caption('These are saved validation figures. No calibrator was fitted, and threshold 0.50 remains fixed. Test performance is never used here to choose another model or threshold.')
