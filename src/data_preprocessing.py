import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder


def load_and_clean(filepath):
    data = pd.read_csv(filepath)
    data.columns = data.columns.str.strip()
    
    data = data.drop_duplicates()
    data = data.dropna()

    return data


def prepare_dataset(filepath, target_column="Attack type"):
    data = load_and_clean(filepath)
    X = data.drop(columns=[target_column, "id"])
    y = data[target_column]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return X, y_encoded, le


def apply_smote(X_train, y_train, random_state=42):
    smote = SMOTE(random_state=random_state)
    return smote.fit_resample(X_train, y_train)
