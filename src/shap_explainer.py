import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# load model + scaler
rf = joblib.load("models/rf_model.pkl")
scaler = joblib.load("models/scaler.pkl")

def generate_shap_plot(uploaded_file):

    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file)


    # remove unwanted
    for col in ["Attack type", "id"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    df.columns = df.columns.str.strip()

    # match training columns
    trained_cols = list(scaler.feature_names_in_)

    for col in trained_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[trained_cols].copy()
    X_scaled = scaler.transform(X)

    # sample for speed
    X_sample = X[:200]

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_sample)

    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)

    plt.savefig("shap_summary.png", bbox_inches="tight", dpi=300)
    plt.close()

    return "shap_summary.png"
