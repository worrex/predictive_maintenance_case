import pandas as pd
import pytest

# Beispiel: Simuliere Feature Engineering
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df["rolling_temp"] = df.groupby("vehicle_id")["engine_temp"].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    df["error_cumsum"] = df.groupby("vehicle_id")["error_code"].cumsum()
    return df

def test_feature_engineering_columns():
    data = {
        "vehicle_id": [1, 1, 1, 2, 2],
        "engine_temp": [85, 87, 83, 90, 92],
        "error_code": [0, 1, 0, 0, 1],
    }
    df = pd.DataFrame(data)
    result = compute_features(df)

    assert "rolling_temp" in result.columns
    assert "error_cumsum" in result.columns
    assert not result["rolling_temp"].isnull().any()
