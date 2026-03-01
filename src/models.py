"""
ML Models Module for Waste to Wealth Project.

Provides training, evaluation, comparison, and loading for SVM and Random Forest.
Each model is a MultiOutputRegressor wrapping the base estimator so it can
predict all four targets (stability, tensile strength, durability, cost reduction)
simultaneously.
"""

import os
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.config import (
    MODELS_DIR,
    SVM_MODEL_PATH,
    RF_MODEL_PATH,
    SVM_PARAMS,
    RF_PARAMS,
    TARGET_COLUMNS,
)


def _ensure_models_dir():
    os.makedirs(MODELS_DIR, exist_ok=True)


# ── Training ─────────────────────────────────────────────────────────────────

def train_svm(X_train, y_train):
    """Train an SVM (RBF kernel) wrapped in MultiOutputRegressor."""
    _ensure_models_dir()
    base = SVR(**SVM_PARAMS)
    model = MultiOutputRegressor(base)

    start = time.time()
    model.fit(X_train, y_train)
    train_time = round(time.time() - start, 2)

    joblib.dump(model, SVM_MODEL_PATH)
    return model, train_time


def train_random_forest(X_train, y_train):
    """Train a Random Forest wrapped in MultiOutputRegressor."""
    _ensure_models_dir()
    base = RandomForestRegressor(**RF_PARAMS)
    model = MultiOutputRegressor(base)

    start = time.time()
    model.fit(X_train, y_train)
    train_time = round(time.time() - start, 2)

    joblib.dump(model, RF_MODEL_PATH)
    return model, train_time


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate a trained model and return per-target metrics + overall."""
    y_pred = model.predict(X_test)
    results = {}

    for i, col in enumerate(TARGET_COLUMNS):
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        results[col] = {"R²": round(r2, 4), "MAE": round(mae, 4), "RMSE": round(rmse, 4)}

    # Overall averages
    overall_r2 = r2_score(y_test, y_pred, multioutput="uniform_average")
    overall_mae = mean_absolute_error(y_test, y_pred, multioutput="uniform_average")
    overall_rmse = np.sqrt(mean_squared_error(y_test, y_pred, multioutput="uniform_average"))
    results["overall"] = {
        "R²": round(overall_r2, 4),
        "MAE": round(overall_mae, 4),
        "RMSE": round(overall_rmse, 4),
    }
    return results


def compare_models(svm_metrics: dict, rf_metrics: dict) -> pd.DataFrame:
    """Return a comparison DataFrame of SVM vs Random Forest metrics."""
    rows = []
    for target in TARGET_COLUMNS + ["overall"]:
        rows.append(
            {
                "Target": target,
                "SVM R²": svm_metrics[target]["R²"],
                "SVM MAE": svm_metrics[target]["MAE"],
                "RF R²": rf_metrics[target]["R²"],
                "RF MAE": rf_metrics[target]["MAE"],
            }
        )
    return pd.DataFrame(rows)


# ── Loading ──────────────────────────────────────────────────────────────────

def load_model(model_name: str = "random_forest"):
    """Load a saved model from disk."""
    path = RF_MODEL_PATH if model_name == "random_forest" else SVM_MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at {path}. Please train the model first."
        )
    return joblib.load(path)
