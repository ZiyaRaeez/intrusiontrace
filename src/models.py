import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


# DECISION TREE
def train_decision_tree(X_train, y_train):
    dt = DecisionTreeClassifier(
        max_depth=15,
        random_state=42
    )
    dt.fit(X_train, y_train)
    return dt


# SVM (Linear)
def train_svm(X_train, y_train):
    svm = LinearSVC(
        C=1.0,
        max_iter=3000,
        random_state=42
    )
    svm.fit(X_train, y_train)
    return svm


# RANDOM FOREST (MAIN MODEL)
def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=80,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    return rf


def evaluate_predictions(y_test, predictions, scores, classes):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )
    y_test_binary = label_binarize(y_test, classes=classes)
    roc_auc = roc_auc_score(
        y_test_binary,
        scores,
        average="weighted",
        multi_class="ovr",
    )

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


def model_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)

    scores = model.decision_function(X)
    scores = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def hybrid_ensemble_predict(rf, dt, svm, X_rf, X_dt, X_svm):
    all_preds = np.vstack([
        rf.predict(X_rf),
        dt.predict(X_dt),
        svm.predict(X_svm),
    ])

    return np.apply_along_axis(
        lambda x: np.bincount(x).argmax(),
        axis=0,
        arr=all_preds
    )


def hybrid_ensemble_scores(rf, dt, svm, X_rf, X_dt, X_svm):
    return np.mean(
        [
            model_scores(rf, X_rf),
            model_scores(dt, X_dt),
            model_scores(svm, X_svm),
        ],
        axis=0,
    )
