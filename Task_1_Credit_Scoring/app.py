import json,joblib,streamlit as st
import pandas as pd
import plotly.express as px
st.set_page_config(page_title="CreditSense AI",page_icon="💳",layout="wide")
st.markdown("<h1>💳 CreditSense <span style='color:#7c3aed'>AI</span></h1>",unsafe_allow_html=True)
st.caption("Explainable Credit Scoring • Model Evaluation • Risk-Aware Demo")
try:
 model=joblib.load("artifacts/credit_model.joblib")
 data=json.load(open("artifacts/metrics.json"))
except:
 st.error("Run `python train.py` first."); st.stop()
m=data["test_metrics"]; cv=data["cv_results"]
c1,c2,c3,c4=st.columns(4)
c1.metric("Test Accuracy",f"{m['accuracy']:.2%}"); c2.metric("F1",f"{m['f1']:.3f}")
c3.metric("ROC-AUC",f"{m['roc_auc']:.3f}"); c4.metric("PR-AUC",f"{m['pr_auc']:.3f}")
st.divider()
tab1,tab2,tab3=st.tabs(["📊 Model Lab","🧠 Explainability","🛡️ Responsible AI"])
with tab1:
 df=pd.DataFrame(cv).T.reset_index(names="Model")
 st.subheader("Cross-validation comparison")
 st.dataframe(df.style.format({c:"{:.3f}" for c in df.columns if c!="Model"}),use_container_width=True)
 st.plotly_chart(px.bar(df,x="Model",y="roc_auc",title="Mean 5-Fold ROC-AUC"),use_container_width=True)
with tab2:
 st.subheader("Why explainability matters")
 st.write("A production version should expose the strongest factors behind a prediction, uncertainty, threshold behavior and data-quality warnings.")
 st.info("For tree models, add SHAP values in the next deployment stage. This demo deliberately avoids inventing explanations that were not calculated.")
with tab3:
 st.warning("Educational use only. Credit decisions require legal, fairness, privacy and human-review controls.")
 st.write("Recommended production controls: subgroup performance checks, drift monitoring, calibration monitoring, audit logs, consent/privacy controls and human review.")
