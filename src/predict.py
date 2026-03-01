"""
Prediction / Inference Module for Waste to Wealth ML Project.

Accepts user inputs (plastic %, plastic type) and returns predicted
road performance metrics using a trained model.
"""

import os
import numpy as np
import joblib

from src.config import (
    PLASTIC_TYPE_MAP,
    SCALER_PATH,
    TARGET_COLUMNS,
)
from src.models import load_model


def _get_default_road_params(plastic_pct: float):
    """
    Estimate default softening point & penetration value from plastic %
    so the user only needs to provide plastic % and type.
    Based on domain relationships encoded in dataset generation.
    """
    softening_point = 45 + 1.8 * plastic_pct
    penetration_value = 80 - 2.5 * plastic_pct
    return softening_point, penetration_value


def predict(
    plastic_pct: float,
    plastic_type: str = "LDPE",
    model_name: str = "random_forest",
) -> dict:
    """
    Make a prediction for given plastic % and type.

    Returns dict with keys:
      marshall_stability, tensile_strength, durability_score,
      cost_reduction_pct, durability_label, strength_improvement,
      recommendation
    """
    # Validate inputs
    if plastic_pct < 0 or plastic_pct > 15:
        raise ValueError("Plastic percentage must be between 0 and 15.")
    if plastic_type not in PLASTIC_TYPE_MAP:
        raise ValueError(f"Plastic type must be one of {list(PLASTIC_TYPE_MAP.keys())}.")

    # Load model and scaler
    model = load_model(model_name)
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Scaler not found. Please train the model first.")
    scaler = joblib.load(SCALER_PATH)

    # Build feature vector
    bitumen_pct = 100 - plastic_pct
    plastic_type_encoded = PLASTIC_TYPE_MAP[plastic_type]
    softening_point, penetration_value = _get_default_road_params(plastic_pct)

    features = np.array(
        [[plastic_pct, bitumen_pct, plastic_type_encoded, softening_point, penetration_value]]
    )
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)[0]

    # Unpack
    result = {}
    for i, col in enumerate(TARGET_COLUMNS):
        result[col] = round(float(prediction[i]), 2)

    # Derived labels
    ds = result["durability_score"]
    if ds >= 75:
        result["durability_label"] = "High"
    elif ds >= 55:
        result["durability_label"] = "Medium"
    else:
        result["durability_label"] = "Low"

    # Strength improvement vs 0% plastic baseline (~2.0 MPa)
    baseline_strength = 2.0
    result["strength_improvement"] = round(
        ((result["tensile_strength"] - baseline_strength) / baseline_strength) * 100, 1
    )

    # Recommendation
    if result["durability_label"] == "High" and result["cost_reduction_pct"] > 10:
        result["recommendation"] = "Highly Recommended ✅"
    elif result["durability_label"] == "Medium":
        result["recommendation"] = "Acceptable ⚠️"
    else:
        result["recommendation"] = "Not Recommended ❌"

    return result
