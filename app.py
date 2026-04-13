import streamlit as st
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import issparse
import warnings
warnings.filterwarnings('ignore')

# Load Models
model = joblib.load('xgb_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
explainer = shap.TreeExplainer(model)

# UI
st.title("🛡️ SQL Injection Detector")
st.write("Enter a SQL query below to check if it is safe or malicious.")

query_text = st.text_area("Enter SQL Query here:")

if st.button("🔍 Detect"):
    if query_text:
        # Prediction
        transformed_query = vectorizer.transform([query_text])
        prediction = model.predict(transformed_query)[0]
        prediction_proba = model.predict_proba(transformed_query)[0][1]

        # SHAP Values
        shap_values = explainer.shap_values(transformed_query)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        if issparse(shap_values):
            shap_values = shap_values.toarray()

        # Result
        if prediction == 1:
            st.error(f"⚠️ MALICIOUS — SQL Injection Detected!")
        else:
            st.success(f"✅ SAFE — Normal Query")

        st.info(f"🎯 Confidence: {round(prediction_proba * 100, 2)}%")

        # Top Features Table
        feature_names = vectorizer.get_feature_names_out()
        df = pd.DataFrame({'ngram': feature_names, 'shap_value': shap_values[0]})
        df['abs_shap'] = df['shap_value'].abs()
        df_sorted = df.sort_values(by='abs_shap', ascending=False).head(10)

        st.subheader("🔎 Top 10 Suspicious Patterns")
        st.dataframe(df_sorted[['ngram', 'shap_value']])

        # SHAP Plot
        st.subheader("📊 SHAP Explanation Chart")
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.bar(shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=None,
            feature_names=feature_names),
            max_display=10, show=False)
        st.pyplot(plt.gcf())
        plt.close()

    else:
        st.warning("⚠️ Please enter a SQL query first!")