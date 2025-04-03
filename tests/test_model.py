from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def generate_dummy_data(n=100):
    df = pd.DataFrame({
        "mean_rpm": np.random.normal(2500, 100, n),
        "std_rpm": np.random.normal(200, 20, n),
        "mean_temp": np.random.normal(85, 5, n),
        "std_temp": np.random.normal(2, 0.5, n),
        "error_count_sum": np.random.randint(0, 5, n),
        "error_count_max": np.random.randint(0, 3, n),
    })
    df["label"] = (df["error_count_sum"] > 2).astype(int)
    return df

def test_model_training():
    df = generate_dummy_data()
    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    assert len(preds) == len(y_test)
