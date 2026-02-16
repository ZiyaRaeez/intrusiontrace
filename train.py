import os
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.data_preprocessing import prepare_data
from src.feature_engineering import apply_scaling, apply_pca, apply_kmeans
from src.models import train_decision_tree, train_svm, train_random_forest, evaluate_model


def main():

    print("\n========== IntrusionTrace Training Started ==========\n")

    # -----------------------------
    # Step 1: Load + preprocess
    # -----------------------------
    X_train, X_test, y_train, y_test, le = prepare_data("data/WSN-DS.csv")

    # -----------------------------
    # Step 2: Scaling
    # -----------------------------
    X_train_scaled, X_test_scaled, scaler = apply_scaling(X_train, X_test)

    # -----------------------------
    # Step 3: PCA
    # -----------------------------
    X_train_pca, X_test_pca, pca = apply_pca(X_train_scaled, X_test_scaled)

    # -----------------------------
    # Step 4: KMeans feature
    # -----------------------------
    X_train_final, X_test_final, kmeans = apply_kmeans(X_train_pca, X_test_pca)

    # =============================
    # MODEL 1: DECISION TREE
    # =============================
    dt = train_decision_tree(X_train, y_train)
    dt_acc, dt_report, dt_cm = evaluate_model(dt, X_test, y_test, "Decision Tree")

    # =============================
    # MODEL 2: SVM + KMEANS
    # =============================
    svm = train_svm(X_train_final, y_train)
    svm_acc, svm_report, svm_cm = evaluate_model(svm, X_test_final, y_test, "SVM + KMeans")

    # =============================
    # MODEL 3: RANDOM FOREST
    # =============================
    rf = train_random_forest(X_train, y_train)
    rf_acc, rf_report, rf_cm = evaluate_model(rf, X_test, y_test, "Random Forest")

    # =========================================================
    # HYBRID ENSEMBLE MODEL (RF + DT + SVM)
    # =========================================================
    print("\n========== Hybrid Ensemble Evaluation ==========\n")

    # Individual predictions
    dt_pred = dt.predict(X_test)
    rf_pred = rf.predict(X_test)
    svm_pred = svm.predict(X_test_final)

    # Stack predictions
    all_preds = np.vstack([dt_pred, rf_pred, svm_pred])

    # Majority voting
    ensemble_pred = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(),
        axis=0,
        arr=all_preds
    )

    # Accuracy
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    print(f"Hybrid Ensemble Accuracy: {ensemble_acc:.4f}")

    # Classification report
    print("\nHybrid Classification Report:\n")
    print(classification_report(y_test, ensemble_pred, target_names=le.classes_))

    # Confusion matrix
    cm = confusion_matrix(y_test, ensemble_pred)
    print("\nHybrid Confusion Matrix:\n", cm)

    # Plot confusion matrix
    plt.figure(figsize=(7,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.title("Hybrid Ensemble Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # =============================
    # CREATE MODELS FOLDER
    # =============================
    os.makedirs("models", exist_ok=True)

    # =============================
    # SAVE MODELS
    # =============================
    joblib.dump(dt, "models/dt_model.pkl")
    joblib.dump(svm, "models/svm_model.pkl")
    joblib.dump(rf, "models/rf_model.pkl")

    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(pca, "models/pca.pkl")
    joblib.dump(kmeans, "models/kmeans.pkl")
    joblib.dump(le, "models/label_encoder.pkl")

    print("\nAll models saved in 'models/' folder")
    print("========== Training Complete ==========\n")


if __name__ == "__main__":
    main()
