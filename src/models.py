from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


def train_decision_tree(X_train, y_train):
    dt = DecisionTreeClassifier(
        max_depth=15,
        random_state=42
    )

    dt.fit(X_train, y_train)
    return dt


def train_svm(X_train, y_train):
    svm = LinearSVC(
        C=1.0,
        max_iter=3000,
        random_state=42
    )

    svm.fit(X_train, y_train)
    return svm


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    return accuracy, report
