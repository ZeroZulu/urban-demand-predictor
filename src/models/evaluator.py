"""Model evaluation metrics and comparison utilities."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error, ignoring zero actuals."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return a dict of regression metrics."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "mape": mape(y_true, y_pred),
        "r2":   float(r2_score(y_true, y_pred)),
    }


def comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """Pretty-print a model comparison DataFrame."""
    df = pd.DataFrame(results).T
    df = df.round({"rmse": 2, "mae": 2, "mape": 2, "r2": 3})
    df.index.name = "model"
    return df.sort_values("rmse")
