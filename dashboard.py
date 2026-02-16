import streamlit as st
import pandas as pd
import time
from src.predict import predict_file
from src.shap_explainer import generate_shap_plot

st.set_page_config(page_title="IntrusionTrace", layout="wide")

st.title("🚨 IntrusionTrace Hybrid Intrusion Detection System")
st.write("AI-Powered Intrusion Detection with Explainable AI")

uploaded = st.file_uploader("Upload WSN CSV file", type=["csv"])

# =========================================================
# AFTER FILE UPLOAD
# =========================================================
if uploaded:

    st.success("Dataset uploaded successfully")

    # reset pointer before reading
    uploaded.seek(0)
    result = predict_file(uploaded)

    # =====================================================
    # KPI OVERVIEW
    # =====================================================
    st.subheader("📊 Intrusion Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", len(result))
    col2.metric("🔴 RED Alerts", (result["Alert"] == "RED").sum())
    col3.metric("🟡 YELLOW Alerts", (result["Alert"] == "YELLOW").sum())
    col4.metric("🟢 GREEN Alerts", (result["Alert"] == "GREEN").sum())

    st.divider()

    # =====================================================
    # CHARTS
    # =====================================================
    colA, colB = st.columns(2)

    with colA:
        st.subheader("Attack Type Distribution")
        st.bar_chart(result["Predicted_Attack"].value_counts())

    with colB:
        st.subheader("Alert Distribution")
        st.bar_chart(result["Alert"].value_counts())

    st.divider()

    # =====================================================
    # HIGH RISK TABLE
    # =====================================================
    st.subheader("🔴 High Risk Intrusions (Top 20)")

    high_risk = result[result["Alert"] == "RED"] \
        .sort_values(by="Risk_%", ascending=False) \
        .head(20)

    st.dataframe(high_risk, use_container_width=True)

    st.divider()

    # =====================================================
    # FULL RESULTS
    # =====================================================
    st.subheader("📋 Full Prediction Results")
    st.dataframe(result, use_container_width=True)

    # =====================================================
    # DOWNLOAD REPORT
    # =====================================================
    st.download_button(
        "⬇ Download Intrusion Report",
        result.to_csv(index=False),
        file_name="intrusiontrace_report.csv"
    )

    st.divider()

    # =====================================================
    # SHAP EXPLAINABLE AI
    # =====================================================
    st.subheader("🧠 Explainable AI — SHAP Feature Importance")
    st.write("Shows which features influenced intrusion predictions")

    if st.button("Generate SHAP Explanation"):
        uploaded.seek(0)
        with st.spinner("Generating SHAP explanation..."):
            img_path = generate_shap_plot(uploaded)
            st.image(img_path, caption="SHAP Feature Importance", use_container_width=True)

else:
    st.info("Upload a WSN dataset CSV file to begin intrusion detection.")
