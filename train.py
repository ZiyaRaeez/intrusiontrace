import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.config import ABLATION_CONFIGS
from src.data_preprocessing import apply_smote, prepare_dataset
from src.feature_engineering import apply_kmeans, apply_pca, apply_scaling
from src.models import (
    evaluate_predictions,
    hybrid_ensemble_predict,
    hybrid_ensemble_scores,
    model_scores,
    train_decision_tree,
    train_random_forest,
    train_svm,
)


DATA_PATH = "data/WSN-DS.csv"
FOLD_RESULTS_PATH = "results/cv_fold_results.csv"
AGGREGATED_RESULTS_PATH = "results/cv_aggregated_results.csv"
N_SPLITS = 10
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")


def build_svm_features(X_train, X_test, config):
    X_train_features, X_test_features, _ = apply_scaling(X_train, X_test)

    if config.use_pca:
        X_train_features, X_test_features, _ = apply_pca(
            X_train_features,
            X_test_features,
        )

    if config.use_kmeans:
        X_train_features, X_test_features, _ = apply_kmeans(
            X_train_features,
            X_test_features,
        )

    return X_train_features, X_test_features


def evaluate_model(model_name, model, X_test, y_test, classes):
    predictions = model.predict(X_test)
    scores = model_scores(model, X_test)
    metrics = evaluate_predictions(y_test, predictions, scores, classes)
    return {"model": model_name, **metrics}


def evaluate_fold(config, fold, X_train, X_test, y_train, y_test, classes):
    if config.use_smote:
        X_train, y_train = apply_smote(X_train, y_train, random_state=42 + fold)

    X_train_svm, X_test_svm = build_svm_features(X_train, X_test, config)
    dt = train_decision_tree(X_train, y_train)
    rf = train_random_forest(X_train, y_train)
    svm = train_svm(X_train_svm, y_train)

    model_results = [
        evaluate_model("Decision Tree", dt, X_test, y_test, classes),
        evaluate_model("Random Forest", rf, X_test, y_test, classes),
        evaluate_model("SVM", svm, X_test_svm, y_test, classes),
    ]

    if config.use_ensemble:
        predictions = hybrid_ensemble_predict(
            rf,
            dt,
            svm,
            X_test,
            X_test,
            X_test_svm,
        )
        scores = hybrid_ensemble_scores(
            rf,
            dt,
            svm,
            X_test,
            X_test,
            X_test_svm,
        )
        metrics = evaluate_predictions(y_test, predictions, scores, classes)
        model_results.append({"model": "Hybrid Ensemble", **metrics})

    metadata = {
        "configuration": config.name,
        "use_pca": config.use_pca,
        "use_smote": config.use_smote,
        "use_kmeans": config.use_kmeans,
        "use_ensemble": config.use_ensemble,
        "fold": fold,
        "train_samples": len(y_train),
        "test_samples": len(y_test),
    }
    return [{**metadata, **result} for result in model_results]


def aggregate_results(fold_results):
    group_columns = [
        "configuration",
        "use_pca",
        "use_smote",
        "use_kmeans",
        "use_ensemble",
        "model",
    ]
    aggregated = fold_results.groupby(group_columns, sort=False)[list(METRICS)].agg(
        ["mean", "std"]
    )
    aggregated.columns = [
        f"{metric}_{statistic}" for metric, statistic in aggregated.columns
    ]
    return aggregated.reset_index()


def save_results(fold_results, fold_path, aggregated_path):
    results_directory = os.path.dirname(fold_path) or os.path.dirname(aggregated_path)
    if results_directory:
        os.makedirs(results_directory, exist_ok=True)

    fold_results.to_csv(fold_path, index=False)
    aggregate_results(fold_results).to_csv(aggregated_path, index=False)


def run_cross_validation(
    data_path=DATA_PATH,
    fold_path=FOLD_RESULTS_PATH,
    aggregated_path=AGGREGATED_RESULTS_PATH,
):
    X, y, label_encoder = prepare_dataset(data_path)
    classes = np.arange(len(label_encoder.classes_))
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    results = []

    for config in ABLATION_CONFIGS:
        print(f"\n========== Running: {config.name} ==========")
        for fold, (train_indices, test_indices) in enumerate(splitter.split(X, y), start=1):
            print(f"Fold {fold}/{N_SPLITS}")
            fold_results = evaluate_fold(
                config,
                fold,
                X.iloc[train_indices],
                X.iloc[test_indices],
                y[train_indices],
                y[test_indices],
                classes,
            )
            results.extend(fold_results)
            save_results(pd.DataFrame(results), fold_path, aggregated_path)

    fold_results_df = pd.DataFrame(results)
    aggregated_results_df = aggregate_results(fold_results_df)
    print(f"\nFold-level results saved to '{fold_path}'")
    print(f"Aggregated results saved to '{aggregated_path}'")
    print(
        aggregated_results_df[
            ["configuration", "model", "accuracy_mean", "f1_mean", "roc_auc_mean"]
        ].to_string(index=False)
    )
    return fold_results_df, aggregated_results_df


def main():
    print("\n========== IntrusionTrace 10-Fold Cross Validation Started ==========\n")
    run_cross_validation()
    print("\n========== Cross Validation Complete ==========\n")


if __name__ == "__main__":
    main()
