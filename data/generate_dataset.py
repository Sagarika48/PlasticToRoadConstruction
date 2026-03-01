"""
Synthetic Dataset Generator for Waste to Wealth ML Project.

Generates three datasets with domain-accurate correlations:
  1. plastic_waste.csv       – plastic type, availability, processing info
  2. bitumen_road_properties.csv – road performance metrics vs plastic %
  3. cost_dataset.csv        – construction cost metrics vs plastic %

Run directly:  python data/generate_dataset.py
"""

import os
import sys
import numpy as np
import pandas as pd

# Allow running as a script from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    DATA_DIR,
    PLASTIC_TYPES,
    DATASET_ROWS,
    RANDOM_STATE,
)

np.random.seed(RANDOM_STATE)


def _ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


# ── 1. Plastic Waste Dataset ────────────────────────────────────────────────
def generate_plastic_waste_dataset(n: int = DATASET_ROWS) -> pd.DataFrame:
    """Plastic type, availability, shredded size, melting point."""
    plastic_type = np.random.choice(PLASTIC_TYPES, size=n)

    # Availability in kg – differs by type
    availability = np.where(
        plastic_type == "LDPE",
        np.random.uniform(500, 2000, n),
        np.where(
            plastic_type == "HDPE",
            np.random.uniform(300, 1500, n),
            np.random.uniform(200, 1000, n),
        ),
    )

    # Shredded particle size (mm)
    shredded_size = np.random.uniform(2.0, 6.0, n)

    # Melting point (°C) – type-dependent
    melting_point = np.where(
        plastic_type == "LDPE",
        np.random.uniform(105, 115, n),
        np.where(
            plastic_type == "HDPE",
            np.random.uniform(120, 135, n),
            np.random.uniform(130, 170, n),
        ),
    )

    df = pd.DataFrame(
        {
            "plastic_type": plastic_type,
            "availability_kg": np.round(availability, 1),
            "shredded_size_mm": np.round(shredded_size, 2),
            "melting_point_c": np.round(melting_point, 1),
        }
    )
    return df


# ── 2. Bitumen & Road Properties Dataset ────────────────────────────────────
def generate_bitumen_road_properties(n: int = DATASET_ROWS) -> pd.DataFrame:
    """
    Road performance metrics as a function of plastic %.
    Domain rules:
      - Marshall stability peaks around 6-8 % plastic
      - Softening point increases with plastic %
      - Penetration value decreases with plastic %
      - Tensile strength peaks at moderate plastic %
      - Durability improves up to ~10 %, then levels off
    """
    plastic_pct = np.random.uniform(0, 15, n)
    bitumen_pct = 100 - plastic_pct

    # Plastic type for each row
    plastic_type = np.random.choice(PLASTIC_TYPES, size=n)
    type_bonus = np.where(
        plastic_type == "HDPE", 0.8, np.where(plastic_type == "PP", 0.5, 0.0)
    )

    noise = lambda scale=1.0: np.random.normal(0, scale, n)

    # Marshall Stability (kN) – peaks at ~7%
    marshall_stability = (
        8
        + 4 * np.exp(-0.5 * ((plastic_pct - 7) / 2.5) ** 2)
        + type_bonus
        + noise(0.4)
    )
    marshall_stability = np.clip(marshall_stability, 5, 16)

    # Softening Point (°C) – increases with plastic %
    softening_point = 45 + 1.8 * plastic_pct + type_bonus * 2 + noise(1.5)
    softening_point = np.clip(softening_point, 42, 80)

    # Penetration Value (mm) – decreases with plastic %
    penetration_value = 80 - 2.5 * plastic_pct - type_bonus * 3 + noise(2.0)
    penetration_value = np.clip(penetration_value, 30, 90)

    # Tensile Strength (MPa) – peaks at ~8%
    tensile_strength = (
        2.0
        + 1.5 * np.exp(-0.5 * ((plastic_pct - 8) / 3) ** 2)
        + type_bonus * 0.3
        + noise(0.15)
    )
    tensile_strength = np.clip(tensile_strength, 1.5, 5.0)

    # Durability score (1-100) – improves up to ~10%
    durability_score = (
        50
        + 30 * (1 - np.exp(-0.3 * plastic_pct))
        + type_bonus * 5
        + noise(3.0)
    )
    durability_score = np.clip(durability_score, 30, 100)

    df = pd.DataFrame(
        {
            "plastic_pct": np.round(plastic_pct, 2),
            "bitumen_pct": np.round(bitumen_pct, 2),
            "plastic_type": plastic_type,
            "marshall_stability": np.round(marshall_stability, 2),
            "softening_point": np.round(softening_point, 2),
            "penetration_value": np.round(penetration_value, 2),
            "tensile_strength": np.round(tensile_strength, 3),
            "durability_score": np.round(durability_score, 2),
        }
    )
    return df


# ── 3. Cost Dataset ─────────────────────────────────────────────────────────
def generate_cost_dataset(n: int = DATASET_ROWS) -> pd.DataFrame:
    """
    Cost metrics vs plastic %.
    Domain rules:
      - Bitumen cost decreases linearly with plastic %
      - Plastic processing cost increases with plastic %
      - Maintenance cost decreases with improved durability
      - Net cost reduction grows with plastic % (to a limit)
    """
    plastic_pct = np.random.uniform(0, 15, n)
    noise = lambda scale=1.0: np.random.normal(0, scale, n)

    # Bitumen cost per km (₹ lakhs) – base ~12, decreases with plastic
    bitumen_cost_per_km = 12 - 0.4 * plastic_pct + noise(0.3)
    bitumen_cost_per_km = np.clip(bitumen_cost_per_km, 4, 14)

    # Plastic processing cost per km (₹ lakhs)
    plastic_processing_cost = 0.2 + 0.15 * plastic_pct + noise(0.1)
    plastic_processing_cost = np.clip(plastic_processing_cost, 0.1, 3.0)

    # Maintenance cost per km per year (₹ lakhs)
    maintenance_cost = 3.0 - 0.15 * plastic_pct + noise(0.2)
    maintenance_cost = np.clip(maintenance_cost, 0.5, 4.0)

    total_cost = bitumen_cost_per_km + plastic_processing_cost + maintenance_cost

    # Cost reduction % compared to 0% plastic baseline (~15 lakhs total)
    baseline = 15.0
    cost_reduction_pct = ((baseline - total_cost) / baseline) * 100
    cost_reduction_pct = np.clip(cost_reduction_pct, 0, 40)

    df = pd.DataFrame(
        {
            "plastic_pct": np.round(plastic_pct, 2),
            "bitumen_cost_per_km_lakhs": np.round(bitumen_cost_per_km, 2),
            "plastic_processing_cost_lakhs": np.round(plastic_processing_cost, 2),
            "maintenance_cost_lakhs": np.round(maintenance_cost, 2),
            "total_cost_lakhs": np.round(total_cost, 2),
            "cost_reduction_pct": np.round(cost_reduction_pct, 2),
        }
    )
    return df


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    _ensure_dir()

    print("Generating Plastic Waste Dataset ...")
    df_plastic = generate_plastic_waste_dataset()
    df_plastic.to_csv(os.path.join(DATA_DIR, "plastic_waste.csv"), index=False)
    print(f"  → {len(df_plastic)} rows saved to data/plastic_waste.csv")

    print("Generating Bitumen & Road Properties Dataset ...")
    df_bitumen = generate_bitumen_road_properties()
    df_bitumen.to_csv(
        os.path.join(DATA_DIR, "bitumen_road_properties.csv"), index=False
    )
    print(f"  → {len(df_bitumen)} rows saved to data/bitumen_road_properties.csv")

    print("Generating Cost Dataset ...")
    df_cost = generate_cost_dataset()
    df_cost.to_csv(os.path.join(DATA_DIR, "cost_dataset.csv"), index=False)
    print(f"  → {len(df_cost)} rows saved to data/cost_dataset.csv")

    print("\n✅ All datasets generated successfully!")


if __name__ == "__main__":
    main()
