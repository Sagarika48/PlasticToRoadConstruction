"""
Data Preprocessing Module for Waste to Wealth ML Project.

Handles loading, cleaning, merging, encoding, scaling, and splitting datasets.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from src.config import (
    BITUMEN_PROPERTIES_CSV,
    COST_CSV,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    PLASTIC_TYPE_MAP,
    TEST_SIZE,
    RANDOM_STATE,
    MODELS_DIR,
    SCALER_PATH,
)


def load_datasets():
    """Load the bitumen road properties and cost datasets."""
    df_bitumen = pd.read_csv(BITUMEN_PROPERTIES_CSV)
    df_cost = pd.read_csv(COST_CSV)
    return df_bitumen, df_cost


def merge_datasets(df_bitumen: pd.DataFrame, df_cost: pd.DataFrame) -> pd.DataFrame:
    """
    Merge bitumen properties with cost data on plastic_pct.
    Uses nearest-value merge because plastic_pct values are continuous.
    """
    df_bitumen = df_bitumen.sort_values("plastic_pct").reset_index(drop=True)
    df_cost = df_cost.sort_values("plastic_pct").reset_index(drop=True)

    # Merge by index (both sorted by plastic_pct, same length)
    merged = pd.concat(
        [
            df_bitumen,
            df_cost[["cost_reduction_pct"]],
        ],
        axis=1,
    )
    return merged


def encode_plastic_type(df: pd.DataFrame) -> pd.DataFrame:
    """Encode plastic_type column to numeric using predefined mapping."""
    df = df.copy()
    df["plastic_type_encoded"] = df["plastic_type"].map(PLASTIC_TYPE_MAP)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and remove duplicates."""
    df = df.copy()
    # Fill numeric NaNs with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    # Drop remaining NaN rows (categorical)
    df = df.dropna()
    # Remove exact duplicates
    df = df.drop_duplicates()
    return df


def prepare_features_targets(df: pd.DataFrame):
    """Extract feature matrix X and target matrix y."""
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMNS].values
    return X, y


def scale_features(X_train, X_test, save_scaler: bool = True):
    """Fit StandardScaler on training set and transform both sets."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if save_scaler:
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)

    return X_train_scaled, X_test_scaled, scaler


def get_processed_data():
    """
    Full pipeline: load → merge → clean → encode → split → scale.
    Returns X_train, X_test, y_train, y_test, scaler, and the full dataframe.
    """
    df_bitumen, df_cost = load_datasets()
    df = merge_datasets(df_bitumen, df_cost)
    df = clean_data(df)
    df = encode_plastic_type(df)

    X, y = prepare_features_targets(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, df
