"""
MongoDB Database Module for Waste to Wealth Project.

Connects to MongoDB Atlas and provides CRUD operations for
storing training runs and prediction results.
"""

import os
from datetime import datetime, timezone

try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False

# ── Connection String ────────────────────────────────────────────────────────
MONGO_URI = os.environ.get(
    "MONGO_URI",
    "mongodb+srv://pdreddy:Dhanu123%40@pdr.nfvdmex.mongodb.net/"
)
DB_NAME = "waste_to_wealth"

# ── Singleton client ─────────────────────────────────────────────────────────
_client = None
_db = None


def get_db():
    """Get (or create) a MongoDB database connection."""
    global _client, _db

    if not PYMONGO_AVAILABLE:
        print("[MongoDB] pymongo not installed. Database features disabled.")
        return None

    if _db is not None:
        return _db

    try:
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # Force a connection check
        _client.admin.command("ping")
        _db = _client[DB_NAME]
        print(f"[MongoDB] ✅ Connected to Atlas cluster — database: {DB_NAME}")
        return _db
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        print(f"[MongoDB] ❌ Connection failed: {e}")
        return None
    except Exception as e:
        print(f"[MongoDB] ❌ Unexpected error: {e}")
        return None


# ── Training Runs ────────────────────────────────────────────────────────────

def save_training_result(model_name: str, metrics: dict, train_time: float):
    """Save a model training run to MongoDB."""
    db = get_db()
    if db is None:
        return None

    doc = {
        "timestamp": datetime.now(timezone.utc),
        "model_name": model_name,
        "train_time_seconds": train_time,
        "overall_r2": metrics.get("overall", {}).get("R²"),
        "overall_mae": metrics.get("overall", {}).get("MAE"),
        "overall_rmse": metrics.get("overall", {}).get("RMSE"),
        "per_target_metrics": {
            k: v for k, v in metrics.items() if k != "overall"
        },
    }

    try:
        result = db.training_runs.insert_one(doc)
        print(f"[MongoDB] ✅ Training result saved (id: {result.inserted_id})")
        return result.inserted_id
    except Exception as e:
        print(f"[MongoDB] ❌ Failed to save training result: {e}")
        return None


def get_training_history(limit: int = 20):
    """Fetch recent training runs from MongoDB."""
    db = get_db()
    if db is None:
        return []

    try:
        cursor = db.training_runs.find(
            {}, {"_id": 0}
        ).sort("timestamp", -1).limit(limit)
        return list(cursor)
    except Exception as e:
        print(f"[MongoDB] ❌ Failed to fetch training history: {e}")
        return []


# ── Predictions ──────────────────────────────────────────────────────────────

def save_prediction(plastic_pct: float, plastic_type: str, model_used: str, result: dict):
    """Save a prediction result to MongoDB."""
    db = get_db()
    if db is None:
        return None

    doc = {
        "timestamp": datetime.now(timezone.utc),
        "input": {
            "plastic_pct": plastic_pct,
            "plastic_type": plastic_type,
            "model_used": model_used,
        },
        "results": {
            "marshall_stability": result.get("marshall_stability"),
            "tensile_strength": result.get("tensile_strength"),
            "durability_score": result.get("durability_score"),
            "durability_label": result.get("durability_label"),
            "cost_reduction_pct": result.get("cost_reduction_pct"),
            "strength_improvement": result.get("strength_improvement"),
            "recommendation": result.get("recommendation"),
        },
    }

    try:
        res = db.predictions.insert_one(doc)
        print(f"[MongoDB] ✅ Prediction saved (id: {res.inserted_id})")
        return res.inserted_id
    except Exception as e:
        print(f"[MongoDB] ❌ Failed to save prediction: {e}")
        return None


def get_prediction_history(limit: int = 50):
    """Fetch recent predictions from MongoDB."""
    db = get_db()
    if db is None:
        return []

    try:
        cursor = db.predictions.find(
            {}, {"_id": 0}
        ).sort("timestamp", -1).limit(limit)
        return list(cursor)
    except Exception as e:
        print(f"[MongoDB] ❌ Failed to fetch prediction history: {e}")
        return []


# ── Dataset Stats ────────────────────────────────────────────────────────────

def save_dataset_info(dataset_name: str, rows: int, columns: int, column_names: list):
    """Save dataset generation info to MongoDB."""
    db = get_db()
    if db is None:
        return None

    doc = {
        "timestamp": datetime.now(timezone.utc),
        "dataset_name": dataset_name,
        "rows": rows,
        "columns": columns,
        "column_names": column_names,
    }

    try:
        res = db.datasets.insert_one(doc)
        print(f"[MongoDB] ✅ Dataset info saved: {dataset_name}")
        return res.inserted_id
    except Exception as e:
        print(f"[MongoDB] ❌ Failed to save dataset info: {e}")
        return None
