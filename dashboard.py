import streamlit as st
import pandas as pd
import time
from src.predict import predict_file, get_feature_columns
from src.shap_explainer import generate_shap_plot

st.set_page_config(page_title="IntrusionTrace", layout="wide")

st.title("🚨 IntrusionTrace Hybrid Intrusion Detection System")
st.write("AI-Powered Intrusion Detection with Explainable AI")

# =========================================================
# SIDEBAR INPUT MODE
# =========================================================
st.sidebar.title("Input Mode")

mode = st.sidebar.radio(
    "Choose how to run detection:",
    ["Upload Dataset", "Manual Entry"]
)

uploaded = None

# =========================================================
# DATASET UPLOAD MODE
# =========================================================
if mode == "Upload Dataset":
    uploaded = st.file_uploader("Upload WSN dataset CSV", type=["csv"])


# =========================================================
# MANUAL ENTRY MODE (PASTE SINGLE ROW)
# =========================================================
if mode == "Manual Entry":

    st.subheader("Real-Time Intrusion Prediction")

    cols = get_feature_columns()

    st.info(f"Paste {len(cols)} feature values separated by comma (same order as dataset)")

    sample = ",".join(["0"] * len(cols))
    st.code(sample[:120] + "...", language="text")

    text = st.text_area("Paste feature values here:")

    if st.button("Predict Intrusion"):

        if text.strip() == "":
            st.warning("Paste values first")
        else:
            try:
                values = [float(x.strip()) for x in text.split(",")]

                if len(values) != len(cols):
                    st.error(f"Expected {len(cols)} values but got {len(values)}")
                else:
                    df = pd.DataFrame([values], columns=cols)

                    result = predict_file(df)

                    st.success("Prediction Complete")
                    st.dataframe(result, use_container_width=True)

            except:
                st.error("Invalid format. Use only numbers separated by commas.")

# =========================================================
# AFTER FILE UPLOAD
# =========================================================
if uploaded:

    st.success("Dataset uploaded successfully")

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
    if mode == "Upload Dataset":
        st.info("Upload a WSN dataset CSV file to begin intrusion detection.")
