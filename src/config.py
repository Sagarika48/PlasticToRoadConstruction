"""
Configuration module for Waste to Wealth ML Project.
Central place for all paths, hyperparameters, and column definitions.
"""

import os

# ── Project Root ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Data Paths ───────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PLASTIC_WASTE_CSV = os.path.join(DATA_DIR, "plastic_waste.csv")
BITUMEN_PROPERTIES_CSV = os.path.join(DATA_DIR, "bitumen_road_properties.csv")
COST_CSV = os.path.join(DATA_DIR, "cost_dataset.csv")

# ── Model Paths ──────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SVM_MODEL_PATH = os.path.join(MODELS_DIR, "svm_model.pkl")
RF_MODEL_PATH = os.path.join(MODELS_DIR, "rf_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

# ── Feature / Target Columns ────────────────────────────────────────────────
FEATURE_COLUMNS = [
    "plastic_pct",
    "bitumen_pct",
    "plastic_type_encoded",
    "softening_point",
    "penetration_value",
]

TARGET_COLUMNS = [
    "marshall_stability",
    "tensile_strength",
    "durability_score",
    "cost_reduction_pct",
]

# ── Plastic Types ────────────────────────────────────────────────────────────
PLASTIC_TYPES = ["LDPE", "HDPE", "PP"]
PLASTIC_TYPE_MAP = {"LDPE": 0, "HDPE": 1, "PP": 2}

# ── Model Hyperparameters ───────────────────────────────────────────────────
SVM_PARAMS = {
    "kernel": "rbf",
    "C": 100,
    "gamma": "scale",
    "epsilon": 0.1,
}

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 12,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
}

# ── Training Config ──────────────────────────────────────────────────────────
TEST_SIZE = 0.2
RANDOM_STATE = 42
DATASET_ROWS = 500
