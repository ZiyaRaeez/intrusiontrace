import pandas as pd
import numpy as np
import joblib

# =============================
# LOAD MODELS
# =============================
rf = joblib.load("models/rf_model.pkl")
dt = joblib.load("models/dt_model.pkl")
svm = joblib.load("models/svm_model.pkl")

scaler = joblib.load("models/scaler.pkl")
pca = joblib.load("models/pca.pkl")
kmeans = joblib.load("models/kmeans.pkl")
le = joblib.load("models/label_encoder.pkl")


# =============================
# ALERT SYSTEM
# =============================
def get_alert(score):
    if score < 30:
        return "GREEN"
    elif score < 70:
        return "YELLOW"
    else:
        return "RED"


priority_map = {
    'Blackhole': 'High',
    'Flooding': 'High',
    'Grayhole': 'Medium',
    'TDMA': 'Medium',
    'Normal': 'Low'
}


# =============================
# MAIN PREDICTION FUNCTION
# (Works for both CSV upload & manual input)
# =============================
def predict_file(input_data):

    # ---------------------------------------
    # ACCEPT BOTH: uploaded file OR dataframe
    # ---------------------------------------
    if isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        df = pd.read_csv(input_data)

    # remove label + id if present
    for col in ["Attack type", "id"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # ------------------------------------------------
    # CLEAN COLUMN NAMES
    # ------------------------------------------------
    df.columns = df.columns.str.strip()

    # ------------------------------------------------
    # FORCE MATCH TRAINING FEATURES
    # ------------------------------------------------
    trained_cols = list(scaler.feature_names_in_)

    # add missing columns if not present
    for col in trained_cols:
        if col not in df.columns:
            df[col] = 0

    # keep only trained columns and correct order
    X = df[trained_cols].copy()

    # -----------------------
    # SCALE
    # -----------------------
    X_scaled = scaler.transform(X)

    # -----------------------
    # PCA
    # -----------------------
    X_pca = pca.transform(X_scaled)

    # -----------------------
    # KMEANS
    # -----------------------
    clusters = kmeans.predict(X_pca)
    X_final = np.column_stack((X_pca, clusters))

    # -----------------------
    # MODEL PREDICTIONS
    # -----------------------
    rf_pred = rf.predict(X)
    dt_pred = dt.predict(X)
    svm_pred = svm.predict(X_final)

    # -----------------------
    # ENSEMBLE VOTING
    # -----------------------
    all_preds = np.vstack([rf_pred, dt_pred, svm_pred])

    ensemble_pred = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(),
        axis=0,
        arr=all_preds
    )

    attack_labels = le.inverse_transform(ensemble_pred)

    # -----------------------
    # RISK SCORE
    # -----------------------
    probs = rf.predict_proba(X)
    risk_scores = np.max(probs, axis=1) * 100

    alerts = [get_alert(r) for r in risk_scores]
    priorities = [priority_map.get(a, "Medium") for a in attack_labels]

    # -----------------------
    # OUTPUT
    # -----------------------
    output = pd.DataFrame({
        "Predicted_Attack": attack_labels,
        "Risk_%": risk_scores.round(2),
        "Alert": alerts,
        "Priority": priorities,
        "Confidence_%": risk_scores.round(2)
    })

    return output


# =============================
# RETURN FEATURE COLUMNS
# (for manual entry dashboard)
# =============================
def get_feature_columns():
    return list(scaler.feature_names_in_)
