import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_and_clean(filepath):
    data = pd.read_csv(filepath)
    data.columns = data.columns.str.strip()
    
    data = data.drop_duplicates()
    data = data.dropna()

    return data


def prepare_data(filepath, target_column="Attack type"):
    data = load_and_clean(filepath)
    
    print("Any NaN:", data.isna().any().any())
    print("Any Inf:", np.isinf(data.select_dtypes(include=[np.number])).any().any())
    print("Max value:", data.select_dtypes(include=[np.number]).max().max())
    print("Min value:", data.select_dtypes(include=[np.number]).min().min())


    X = data.drop(columns=[target_column, "id"])
    y = data[target_column]

    print(data.columns)


    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.25,
        random_state=42,
        stratify=y_encoded
    )

    return X_train, X_test, y_train, y_test, le
