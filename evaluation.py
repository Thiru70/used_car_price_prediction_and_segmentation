"""
evaluation.py
-------------
Centralised evaluation metrics for regression models.
Import these helpers in app.py to compare model performance.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_mae(y_true, y_pred) -> float:
    """Mean Absolute Error — average absolute difference between predictions and actuals."""
    return round(float(mean_absolute_error(y_true, y_pred)), 2)


def calculate_rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error — penalises large errors more heavily."""
    mse = mean_squared_error(y_true, y_pred)
    return round(float(np.sqrt(mse)), 2)


def calculate_r2(y_true, y_pred) -> float:
    """R² Score — proportion of variance explained by the model (1.0 = perfect)."""
    return round(float(r2_score(y_true, y_pred)), 4)


def evaluate_model(model_name: str, y_true, y_pred) -> dict:
    """
    Evaluate a single model and return a results dictionary.
    Args:
        model_name : human-readable name for the model
        y_true     : actual target values
        y_pred     : predicted target values
    Returns:
        dict with keys: model, MAE, RMSE, R2
    """
    return {
        "model": model_name,
        "MAE":   calculate_mae(y_true, y_pred),
        "RMSE":  calculate_rmse(y_true, y_pred),
        "R2":    calculate_r2(y_true, y_pred),
    }


def compare_models(results: list[dict]) -> list[dict]:
    """
    Sort a list of model-result dicts by R² score (descending).
    Makes it easy to rank models from best to worst.
    """
    return sorted(results, key=lambda r: r["R2"], reverse=True)
