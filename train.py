from src.data_preprocessing import prepare_data
from src.feature_engineering import apply_scaling, apply_pca, apply_kmeans
from src.models import train_decision_tree, train_svm, evaluate_model


def main():

    # Step 1: Preprocessing
    X_train, X_test, y_train, y_test, le = prepare_data("data/WSN-DS.csv")

    # Step 2: Scaling
    X_train_scaled, X_test_scaled, scaler = apply_scaling(X_train, X_test)


    X_train_pca, X_test_pca, pca = apply_pca(X_train_scaled, X_test_scaled)


    X_train_final, X_test_final, kmeans = apply_kmeans(X_train_pca, X_test_pca)

    dt = train_decision_tree(X_train, y_train)
    dt_acc, dt_report = evaluate_model(dt, X_test, y_test)

    print("\n===== DECISION TREE =====")
    print("Accuracy:", dt_acc)
    print(dt_report)

    svm = train_svm(X_train_final, y_train)
    svm_acc, svm_report = evaluate_model(svm, X_test_final, y_test)

    print("\n===== SVM + KMEANS =====")
    print("Accuracy:", svm_acc)
    print(svm_report)


if __name__ == "__main__":
    main()
