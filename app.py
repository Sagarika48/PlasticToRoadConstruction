"""
Waste to Wealth – Streamlit Application
========================================
A Machine Learning-based prediction system for optimal plastic waste
utilization in road construction.

Run:  streamlit run app.py
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime

# ── Terminal Logger Setup ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("WasteToWealth")


def log_to_terminal(title: str, data: dict = None, message: str = None):
    """Print structured output to the terminal."""
    separator = "═" * 60
    print(f"\n{separator}")
    print(f"  {title}")
    print(f"{separator}")
    if message:
        print(f"  {message}")
    if data:
        for key, value in data.items():
            print(f"  {key:.<35s} {value}")
    print(f"{separator}\n")

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    PLASTIC_WASTE_CSV,
    BITUMEN_PROPERTIES_CSV,
    COST_CSV,
    PLASTIC_TYPES,
    TARGET_COLUMNS,
    MODELS_DIR,
    SVM_MODEL_PATH,
    RF_MODEL_PATH,
    DATA_DIR,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Page Config
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Waste to Wealth – ML Predictor",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# Custom CSS – Premium Dark Theme
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
<style>
/* ── Import Google Font ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root Variables ─────────────────────────────────────────────── */
:root {
    --bg-primary: #0f1117;
    --bg-secondary: #1a1d29;
    --bg-card: #1e2130;
    --accent-green: #00d68f;
    --accent-blue: #3366ff;
    --accent-purple: #9b59b6;
    --accent-orange: #f5a623;
    --text-primary: #e8eaf6;
    --text-secondary: #9ca3af;
    --border-color: #2d3148;
    --gradient-1: linear-gradient(135deg, #00d68f 0%, #3366ff 100%);
    --gradient-2: linear-gradient(135deg, #9b59b6 0%, #3366ff 100%);
    --shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

/* ── Global ─────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif !important;
}

.main .block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

/* ── Metric Cards ───────────────────────────────────────────────── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: var(--shadow);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.4);
}
.metric-card .value {
    font-size: 2.2rem;
    font-weight: 700;
    background: var(--gradient-1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-card .label {
    font-size: 0.85rem;
    color: var(--text-secondary);
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Section Headers ────────────────────────────────────────────── */
.section-header {
    font-size: 1.8rem;
    font-weight: 700;
    margin: 2rem 0 1rem 0;
    background: var(--gradient-1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ── Result Box ─────────────────────────────────────────────────── */
.result-box {
    background: linear-gradient(145deg, #1a2235 0%, #162030 100%);
    border: 1px solid #2d4a3e;
    border-left: 4px solid var(--accent-green);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
}
.result-box h3 {
    margin: 0 0 0.5rem 0;
    color: var(--accent-green);
}
.result-box .big-val {
    font-size: 2rem;
    font-weight: 800;
    color: #fff;
}

/* ── Info Card ──────────────────────────────────────────────────── */
.info-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 1.5rem;
    margin: 0.5rem 0;
    box-shadow: var(--shadow);
}
.info-card h4 {
    margin: 0 0 0.5rem 0;
    color: var(--accent-blue);
}

/* ── Sidebar ────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #141824 0%, #0f1117 100%);
}
[data-testid="stSidebar"] .stRadio > label {
    font-weight: 600;
}

/* ── Buttons ────────────────────────────────────────────────────── */
.stButton > button {
    background: var(--gradient-1) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: opacity 0.2s ease !important;
}
.stButton > button:hover {
    opacity: 0.85 !important;
}

/* ── Dataframe styling ──────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}
</style>
""",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper — check if datasets exist
# ═══════════════════════════════════════════════════════════════════════════════
def datasets_exist():
    return all(
        os.path.exists(p)
        for p in [PLASTIC_WASTE_CSV, BITUMEN_PROPERTIES_CSV, COST_CSV]
    )


def models_trained():
    return os.path.exists(SVM_MODEL_PATH) and os.path.exists(RF_MODEL_PATH)


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar Navigation
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ♻️ Waste to Wealth")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "📊 Dataset Explorer", "🤖 Model Training", "🔮 Prediction", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<p style='color:#9ca3af;font-size:0.8rem;text-align:center;'>"
        "ML-Powered Road Construction<br>Optimization System</p>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Home
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown(
        "<h1 style='text-align:center;'>"
        "<span style='background:linear-gradient(135deg,#00d68f,#3366ff);"
        "-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>"
        "♻️ Waste to Wealth</span></h1>"
        "<p style='text-align:center;color:#9ca3af;font-size:1.1rem;'>"
        "Plastic Waste Utilization in Road Construction using Machine Learning</p>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Key benefit cards
    cols = st.columns(4)
    benefits = [
        ("🛣️", "Road Durability", "Improved lifespan with optimal plastic mix"),
        ("💰", "Cost Reduction", "Up to 20% savings on construction costs"),
        ("💪", "Strength", "Enhanced load-bearing capacity of roads"),
        ("🌿", "Sustainability", "Eco-friendly plastic waste management"),
    ]
    for col, (icon, title, desc) in zip(cols, benefits):
        with col:
            st.markdown(
                f"""<div class='metric-card'>
                    <div class='value'>{icon}</div>
                    <div style='font-size:1.1rem;font-weight:600;color:#e8eaf6;margin:0.5rem 0;'>{title}</div>
                    <div class='label' style='text-transform:none;letter-spacing:0;'>{desc}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # How it works
    st.markdown("<div class='section-header'>How It Works</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    steps = [
        ("1️⃣", "Generate Data", "Create synthetic datasets with domain-accurate correlations"),
        ("2️⃣", "Train Models", "Train SVM & Random Forest regressors on road metrics"),
        ("3️⃣", "Compare", "Evaluate and compare model accuracy (R², MAE, RMSE)"),
        ("4️⃣", "Predict", "Input plastic % to get durability, strength & cost predictions"),
    ]
    for col, (num, title, desc) in zip([c1, c2, c3, c4], steps):
        with col:
            st.markdown(
                f"""<div class='info-card'>
                    <h4>{num} {title}</h4>
                    <p style='color:#9ca3af;font-size:0.9rem;'>{desc}</p>
                </div>""",
                unsafe_allow_html=True,
            )

    # Quick dataset generation
    st.markdown("<br>", unsafe_allow_html=True)
    if not datasets_exist():
        st.warning("⚠️ Datasets not found. Generate them to get started.")
        if st.button("🚀 Generate Datasets", key="home_gen"):
            with st.spinner("Generating synthetic datasets..."):
                from data.generate_dataset import main as gen_main
                gen_main()
            log_to_terminal("DATASET GENERATION", message="✅ All 3 datasets generated (500 rows each)")
            st.success("✅ Datasets generated successfully!")
            st.rerun()
    else:
        st.success("✅ Datasets are ready. Navigate to **Dataset Explorer** or **Model Training**.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Dataset Explorer
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dataset Explorer":
    st.markdown("<div class='section-header'>📊 Dataset Explorer</div>", unsafe_allow_html=True)

    if not datasets_exist():
        st.warning("⚠️ Datasets not found. Please generate them from the Home page first.")
        if st.button("🚀 Generate Datasets"):
            with st.spinner("Generating synthetic datasets..."):
                from data.generate_dataset import main as gen_main
                gen_main()
            log_to_terminal("DATASET GENERATION", message="✅ All 3 datasets generated (500 rows each)")
            st.success("✅ Done!")
            st.rerun()
    else:
        tab1, tab2, tab3 = st.tabs([
            "🧪 Plastic Waste", "🛣️ Bitumen & Road Properties", "💰 Cost Analysis"
        ])

        with tab1:
            df = pd.read_csv(PLASTIC_WASTE_CSV)
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Records", len(df))
            c2.metric("Plastic Types", df["plastic_type"].nunique())
            c3.metric("Avg Availability (kg)", f"{df['availability_kg'].mean():.0f}")
            st.dataframe(df, use_container_width=True, height=350)

            st.markdown("#### Distribution by Plastic Type")
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            fig.patch.set_facecolor('#0f1117')
            for ax in axes:
                ax.set_facecolor('#1a1d29')
                ax.tick_params(colors='#9ca3af')
                ax.xaxis.label.set_color('#9ca3af')
                ax.yaxis.label.set_color('#9ca3af')

            colors = ['#00d68f', '#3366ff', '#9b59b6']
            df["plastic_type"].value_counts().plot.bar(ax=axes[0], color=colors)
            axes[0].set_title("Count by Type", color='#e8eaf6')
            axes[0].set_ylabel("Count")

            for i, pt in enumerate(PLASTIC_TYPES):
                subset = df[df["plastic_type"] == pt]["availability_kg"]
                axes[1].hist(subset, bins=20, alpha=0.6, label=pt, color=colors[i])
            axes[1].set_title("Availability Distribution", color='#e8eaf6')
            axes[1].set_xlabel("Availability (kg)")
            axes[1].legend()
            plt.tight_layout()
            st.pyplot(fig)

        with tab2:
            df = pd.read_csv(BITUMEN_PROPERTIES_CSV)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Records", len(df))
            c2.metric("Avg Stability (kN)", f"{df['marshall_stability'].mean():.2f}")
            c3.metric("Avg Tensile (MPa)", f"{df['tensile_strength'].mean():.3f}")
            c4.metric("Avg Durability", f"{df['durability_score'].mean():.1f}")
            st.dataframe(df, use_container_width=True, height=350)

            st.markdown("#### Properties vs Plastic %")
            fig, axes = plt.subplots(2, 2, figsize=(14, 8))
            fig.patch.set_facecolor('#0f1117')
            props = [
                ("marshall_stability", "Marshall Stability (kN)", '#00d68f'),
                ("softening_point", "Softening Point (°C)", '#3366ff'),
                ("tensile_strength", "Tensile Strength (MPa)", '#9b59b6'),
                ("durability_score", "Durability Score", '#f5a623'),
            ]
            for ax, (col, title, color) in zip(axes.flatten(), props):
                ax.set_facecolor('#1a1d29')
                ax.scatter(df["plastic_pct"], df[col], alpha=0.3, s=8, color=color)
                # trend line
                z = np.polyfit(df["plastic_pct"], df[col], 3)
                p = np.poly1d(z)
                x_smooth = np.linspace(0, 15, 100)
                ax.plot(x_smooth, p(x_smooth), color=color, linewidth=2)
                ax.set_title(title, color='#e8eaf6', fontsize=11)
                ax.set_xlabel("Plastic %", color='#9ca3af')
                ax.tick_params(colors='#9ca3af')
            plt.tight_layout()
            st.pyplot(fig)

        with tab3:
            df = pd.read_csv(COST_CSV)
            c1, c2, c3 = st.columns(3)
            c1.metric("Records", len(df))
            c2.metric("Avg Total Cost (₹L/km)", f"{df['total_cost_lakhs'].mean():.2f}")
            c3.metric("Avg Cost Reduction", f"{df['cost_reduction_pct'].mean():.1f}%")
            st.dataframe(df, use_container_width=True, height=350)

            st.markdown("#### Cost vs Plastic %")
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.patch.set_facecolor('#0f1117')
            for ax in axes:
                ax.set_facecolor('#1a1d29')
                ax.tick_params(colors='#9ca3af')

            ax = axes[0]
            for col, color, label in [
                ("bitumen_cost_per_km_lakhs", "#3366ff", "Bitumen Cost"),
                ("plastic_processing_cost_lakhs", "#f5a623", "Plastic Processing"),
                ("maintenance_cost_lakhs", "#9b59b6", "Maintenance"),
            ]:
                ax.scatter(df["plastic_pct"], df[col], alpha=0.2, s=8, color=color)
                z = np.polyfit(df["plastic_pct"], df[col], 2)
                p = np.poly1d(z)
                x_s = np.linspace(0, 15, 100)
                ax.plot(x_s, p(x_s), color=color, linewidth=2, label=label)
            ax.set_title("Cost Components vs Plastic %", color='#e8eaf6')
            ax.set_xlabel("Plastic %", color='#9ca3af')
            ax.set_ylabel("Cost (₹ Lakhs/km)", color='#9ca3af')
            ax.legend()

            ax = axes[1]
            ax.scatter(df["plastic_pct"], df["cost_reduction_pct"], alpha=0.3, s=8, color='#00d68f')
            z = np.polyfit(df["plastic_pct"], df["cost_reduction_pct"], 2)
            p = np.poly1d(z)
            x_s = np.linspace(0, 15, 100)
            ax.plot(x_s, p(x_s), color='#00d68f', linewidth=2)
            ax.set_title("Cost Reduction % vs Plastic %", color='#e8eaf6')
            ax.set_xlabel("Plastic %", color='#9ca3af')
            ax.set_ylabel("Cost Reduction %", color='#9ca3af')
            plt.tight_layout()
            st.pyplot(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Model Training
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Training":
    st.markdown("<div class='section-header'>🤖 Model Training</div>", unsafe_allow_html=True)

    if not datasets_exist():
        st.warning("⚠️ Datasets not found. Please generate them from the Home page first.")
        st.stop()

    st.markdown(
        "<div class='info-card'>"
        "<h4>Select Algorithm & Train</h4>"
        "<p style='color:#9ca3af;'>Choose one or both ML algorithms to train on the road "
        "construction dataset. Models predict: Marshall Stability, Tensile Strength, "
        "Durability Score, and Cost Reduction %.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    train_svm_flag = col1.checkbox("Support Vector Machine (SVM)", value=True)
    train_rf_flag = col2.checkbox("Random Forest", value=True)

    if st.button("🚀 Train Selected Models", use_container_width=True):
        from src.preprocessing import get_processed_data
        from src.models import train_svm, train_random_forest, evaluate_model, compare_models

        with st.spinner("Preprocessing data..."):
            X_train, X_test, y_train, y_test, scaler, df = get_processed_data()
        log_to_terminal("DATA PREPROCESSING", {
            "Training samples": X_train.shape[0],
            "Test samples": X_test.shape[0],
            "Features": X_train.shape[1],
        })
        st.success(f"✅ Data preprocessed: {X_train.shape[0]} train / {X_test.shape[0]} test samples")

        svm_metrics = None
        rf_metrics = None

        if train_svm_flag:
            with st.spinner("Training SVM model..."):
                svm_model, svm_time = train_svm(X_train, y_train)
                svm_metrics = evaluate_model(svm_model, X_test, y_test)
            log_to_terminal("SVM MODEL TRAINING", {
                "Training Time": f"{svm_time}s",
                "Overall R²": svm_metrics['overall']['R²'],
                "Overall MAE": svm_metrics['overall']['MAE'],
                "Overall RMSE": svm_metrics['overall']['RMSE'],
                **{f"{t} R²": svm_metrics[t]['R²'] for t in TARGET_COLUMNS},
            })
            st.success(f"✅ SVM trained in {svm_time}s — Overall R²: {svm_metrics['overall']['R²']}")

        if train_rf_flag:
            with st.spinner("Training Random Forest model..."):
                rf_model, rf_time = train_random_forest(X_train, y_train)
                rf_metrics = evaluate_model(rf_model, X_test, y_test)
            log_to_terminal("RANDOM FOREST MODEL TRAINING", {
                "Training Time": f"{rf_time}s",
                "Overall R²": rf_metrics['overall']['R²'],
                "Overall MAE": rf_metrics['overall']['MAE'],
                "Overall RMSE": rf_metrics['overall']['RMSE'],
                **{f"{t} R²": rf_metrics[t]['R²'] for t in TARGET_COLUMNS},
            })
            st.success(f"✅ Random Forest trained in {rf_time}s — Overall R²: {rf_metrics['overall']['R²']}")

        # ── Display Metrics ──────────────────────────────────────────────
        st.markdown("<div class='section-header'>📈 Model Performance</div>", unsafe_allow_html=True)

        if svm_metrics:
            st.markdown("#### SVM Results")
            svm_df = pd.DataFrame(svm_metrics).T
            svm_df.index.name = "Target"
            st.dataframe(svm_df.style.format("{:.4f}"), use_container_width=True)

        if rf_metrics:
            st.markdown("#### Random Forest Results")
            rf_df = pd.DataFrame(rf_metrics).T
            rf_df.index.name = "Target"
            st.dataframe(rf_df.style.format("{:.4f}"), use_container_width=True)

        # Comparison chart
        if svm_metrics and rf_metrics:
            st.markdown("<div class='section-header'>🔍 Model Comparison</div>", unsafe_allow_html=True)
            comp_df = compare_models(svm_metrics, rf_metrics)
            st.dataframe(comp_df, use_container_width=True)

            # Visual comparison
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.patch.set_facecolor('#0f1117')
            for ax in axes:
                ax.set_facecolor('#1a1d29')
                ax.tick_params(colors='#9ca3af')

            targets = TARGET_COLUMNS
            x = np.arange(len(targets))
            width = 0.35

            svm_r2 = [svm_metrics[t]["R²"] for t in targets]
            rf_r2 = [rf_metrics[t]["R²"] for t in targets]
            axes[0].bar(x - width / 2, svm_r2, width, label="SVM", color="#3366ff", alpha=0.85)
            axes[0].bar(x + width / 2, rf_r2, width, label="Random Forest", color="#00d68f", alpha=0.85)
            axes[0].set_xticks(x)
            axes[0].set_xticklabels([t.replace("_", "\n") for t in targets], fontsize=8, color='#9ca3af')
            axes[0].set_ylabel("R² Score", color='#9ca3af')
            axes[0].set_title("R² Score Comparison", color='#e8eaf6')
            axes[0].legend()
            axes[0].set_ylim(0, 1.1)

            svm_mae = [svm_metrics[t]["MAE"] for t in targets]
            rf_mae = [rf_metrics[t]["MAE"] for t in targets]
            axes[1].bar(x - width / 2, svm_mae, width, label="SVM", color="#3366ff", alpha=0.85)
            axes[1].bar(x + width / 2, rf_mae, width, label="Random Forest", color="#00d68f", alpha=0.85)
            axes[1].set_xticks(x)
            axes[1].set_xticklabels([t.replace("_", "\n") for t in targets], fontsize=8, color='#9ca3af')
            axes[1].set_ylabel("MAE", color='#9ca3af')
            axes[1].set_title("MAE Comparison", color='#e8eaf6')
            axes[1].legend()
            plt.tight_layout()
            st.pyplot(fig)

    # Show existing model status
    st.markdown("---")
    st.markdown("#### 📦 Saved Models")
    m1, m2 = st.columns(2)
    m1.metric("SVM Model", "✅ Saved" if os.path.exists(SVM_MODEL_PATH) else "❌ Not trained")
    m2.metric("RF Model", "✅ Saved" if os.path.exists(RF_MODEL_PATH) else "❌ Not trained")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Prediction
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Prediction":
    st.markdown("<div class='section-header'>🔮 Prediction Interface</div>", unsafe_allow_html=True)

    if not models_trained():
        st.warning("⚠️ Models not trained yet. Please go to **Model Training** page first.")
        st.stop()

    st.markdown(
        "<div class='info-card'>"
        "<h4>Enter Parameters</h4>"
        "<p style='color:#9ca3af;'>Provide the plastic percentage and type to predict "
        "road performance metrics.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        plastic_pct = st.slider(
            "Plastic Percentage (%)",
            min_value=0.0,
            max_value=15.0,
            value=6.0,
            step=0.5,
            help="Percentage of plastic waste to mix with bitumen",
        )
    with col2:
        plastic_type = st.selectbox(
            "Plastic Type",
            PLASTIC_TYPES,
            help="LDPE (carry bags), HDPE (milk covers), PP (food wrappers)",
        )
    with col3:
        model_choice = st.selectbox(
            "ML Model",
            ["Random Forest", "SVM"],
        )

    if st.button("🔮 Predict Road Performance", use_container_width=True):
        from src.predict import predict

        model_key = "random_forest" if model_choice == "Random Forest" else "svm"

        with st.spinner("Making prediction..."):
            try:
                result = predict(plastic_pct, plastic_type, model_key)
            except Exception as e:
                log_to_terminal("PREDICTION ERROR", message=str(e))
                st.error(f"Prediction error: {e}")
                st.stop()

        # ── Log prediction to terminal ───────────────────────────────
        si = result['strength_improvement']
        sign = "+" if si >= 0 else ""
        log_to_terminal(f"PREDICTION RESULT — {plastic_pct}% {plastic_type} ({model_choice})", {
            "Marshall Stability (kN)": result['marshall_stability'],
            "Tensile Strength (MPa)": result['tensile_strength'],
            "Durability Score": result['durability_score'],
            "Durability Level": result['durability_label'],
            "Cost Reduction (%)": result['cost_reduction_pct'],
            "Strength Improvement (%)": f"{sign}{si}",
            "Recommendation": result['recommendation'],
        })

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Result Cards ─────────────────────────────────────────────
        st.markdown(
            f"<h3 style='text-align:center;color:#e8eaf6;'>"
            f"Results for <span style='color:#00d68f;'>{plastic_pct}% {plastic_type}</span> "
            f"plastic mix</h3>",
            unsafe_allow_html=True,
        )

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.markdown(
                f"""<div class='metric-card'>
                    <div class='value'>{result['durability_label']}</div>
                    <div class='label'>Durability</div>
                    <div style='color:#9ca3af;font-size:0.85rem;'>Score: {result['durability_score']}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with r2:
            st.markdown(
                f"""<div class='metric-card'>
                    <div class='value'>{result['cost_reduction_pct']}%</div>
                    <div class='label'>Cost Reduction</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with r3:
            si = result.get('strength_improvement', result['strength_improvement'])
            sign = "+" if si >= 0 else ""
            st.markdown(
                f"""<div class='metric-card'>
                    <div class='value'>{sign}{si}%</div>
                    <div class='label'>Strength Improvement</div>
                    <div style='color:#9ca3af;font-size:0.85rem;'>Tensile: {result['tensile_strength']} MPa</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with r4:
            st.markdown(
                f"""<div class='metric-card'>
                    <div class='value'>{result['marshall_stability']}</div>
                    <div class='label'>Marshall Stability (kN)</div>
                </div>""",
                unsafe_allow_html=True,
            )

        # Recommendation
        st.markdown(
            f"""<div class='result-box'>
                <h3>Recommendation</h3>
                <div class='big-val'>{result['recommendation']}</div>
                <p style='color:#9ca3af;margin-top:0.5rem;'>
                    Optimal Plastic Content: <strong>{plastic_pct}%</strong> &nbsp;|&nbsp;
                    Durability: <strong>{result['durability_label']}</strong> &nbsp;|&nbsp;
                    Cost Reduction: <strong>{result['cost_reduction_pct']}%</strong> &nbsp;|&nbsp;
                    Strength Improvement: <strong>{sign}{si}%</strong>
                </p>
            </div>""",
            unsafe_allow_html=True,
        )

        # Detailed table
        st.markdown("#### 📋 Detailed Predictions")
        detail_df = pd.DataFrame(
            {
                "Metric": [
                    "Marshall Stability (kN)",
                    "Tensile Strength (MPa)",
                    "Durability Score",
                    "Cost Reduction (%)",
                    "Durability Level",
                    "Strength Improvement (%)",
                ],
                "Value": [
                    result["marshall_stability"],
                    result["tensile_strength"],
                    result["durability_score"],
                    result["cost_reduction_pct"],
                    result["durability_label"],
                    f"{sign}{si}",
                ],
            }
        )
        st.dataframe(detail_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: About
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown("<div class='section-header'>ℹ️ About This Project</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class='info-card'>
        <h4>🎯 Project Goal</h4>
        <p style='color:#9ca3af;'>
        Develop a Machine Learning–based prediction system that determines the optimal
        percentage of plastic waste to mix with bitumen in road construction — reducing
        cost, improving durability, increasing strength, and promoting sustainable waste management.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class='info-card'>
            <h4>🧪 Suitable Plastic Types</h4>
            <table style='color:#e8eaf6;width:100%;'>
                <tr><td><strong>LDPE</strong></td><td>Carry bags, packaging film</td></tr>
                <tr><td><strong>HDPE</strong></td><td>Milk covers, bottles</td></tr>
                <tr><td><strong>PP</strong></td><td>Food wrappers, containers</td></tr>
            </table>
            <br>
            <p style='color:#f5a623;'>⚠️ PVC is NOT suitable (toxic emissions when heated)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class='info-card'>
            <h4>🔬 Process</h4>
            <ol style='color:#9ca3af;'>
                <li>Plastic waste is collected and sorted</li>
                <li>Shredded to 2-6 mm particles</li>
                <li>Mixed with hot aggregates (160-170°C)</li>
                <li>Bitumen is added to the coated aggregates</li>
                <li>Road is laid using conventional methods</li>
            </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class='info-card'>
            <h4>🤖 ML Models Used</h4>
            <p style='color:#9ca3af;'><strong style='color:#3366ff;'>Support Vector Machine (SVM)</strong><br>
            RBF kernel regressor wrapped in MultiOutputRegressor. Good for capturing
            non-linear relationships in moderate datasets.</p>
            <br>
            <p style='color:#9ca3af;'><strong style='color:#00d68f;'>Random Forest</strong><br>
            Ensemble of 200 decision trees. Robust to outliers with built-in feature
            importance estimation.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class='info-card'>
            <h4>📊 Predicted Metrics</h4>
            <ul style='color:#9ca3af;'>
                <li><strong>Marshall Stability</strong> — Road strength indicator (kN)</li>
                <li><strong>Tensile Strength</strong> — Load bearing capacity (MPa)</li>
                <li><strong>Durability Score</strong> — Expected lifespan (1-100)</li>
                <li><strong>Cost Reduction</strong> — Savings vs traditional bitumen (%)</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class='info-card'>
        <h4>⚙️ Technology Stack</h4>
        <p style='color:#9ca3af;'>
        <strong>Backend:</strong> Python, Pandas, NumPy, Scikit-learn &nbsp;|&nbsp;
        <strong>Frontend:</strong> Streamlit &nbsp;|&nbsp;
        <strong>Models:</strong> SVM (RBF), Random Forest &nbsp;|&nbsp;
        <strong>Serialization:</strong> Joblib (.pkl)
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
