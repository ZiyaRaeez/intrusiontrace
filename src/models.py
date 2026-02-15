from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np


# ===============================
# DECISION TREE
# ===============================
def train_decision_tree(X_train, y_train):
    dt = DecisionTreeClassifier(
        max_depth=15,
        random_state=42
    )
    dt.fit(X_train, y_train)
    return dt


# ===============================
# SVM (Linear)
# ===============================
def train_svm(X_train, y_train):
    svm = LinearSVC(
        C=1.0,
        max_iter=3000,
        random_state=42
    )
    svm.fit(X_train, y_train)
    return svm


# ===============================
# RANDOM FOREST (MAIN MODEL)
# ===============================
def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=80,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    return rf


# ===============================
# EVALUATION FUNCTION
# ===============================
def evaluate_model(model, X_test, y_test, model_name="Model"):

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    print(f"\n===== {model_name} RESULTS =====")
    print("Accuracy:", acc)
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", cm)

    return acc, report, cm


# ===============================
# ENSEMBLE PREDICTION (OPTIONAL)
# ===============================
def hybrid_ensemble_predict(rf, dt, svm, X_rf, X_dt, X_svm):
    """
    Majority voting ensemble
    """

    rf_pred = rf.predict(X_rf)
    dt_pred = dt.predict(X_dt)
    svm_pred = svm.predict(X_svm)

    all_preds = np.vstack([rf_pred, dt_pred, svm_pred])

    ensemble_pred = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(),
        axis=0,
        arr=all_preds
    )

    return ensemble_pred
