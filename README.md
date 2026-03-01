# ♻️ Waste to Wealth

**Plastic Waste Utilization in Road Construction using Machine Learning**

An ML-powered prediction system that determines the optimal percentage of plastic waste to mix with bitumen in road construction — reducing cost, improving durability, increasing strength, and promoting sustainable waste management.

---

## 🚀 Features

- **ML Models** — SVM & Random Forest regressors predicting 4 road performance metrics simultaneously
- **Interactive UI** — Streamlit app with 6 pages: Home, Dataset Explorer, Model Training, Prediction, History, About
- **Dataset Generation** — Synthetic data with domain-accurate correlations (stability peaks at 6-8% plastic)
- **Model Comparison** — Side-by-side R², MAE, RMSE comparison charts
- **MongoDB Atlas** — All training runs and predictions are persisted to cloud database
- **Chart Saving** — All generated charts auto-saved as PNG images to `results/` folder
- **Terminal Logging** — Every UI action is mirrored to the terminal with structured output
- **Premium Dark Theme** — Gradient cards, trend-line charts, hover animations

---

## 📊 Predicted Metrics

| Metric | Description |
|--------|-------------|
| Marshall Stability (kN) | Road strength indicator |
| Tensile Strength (MPa) | Load bearing capacity |
| Durability Score (1-100) | Expected lifespan |
| Cost Reduction (%) | Savings vs traditional bitumen |

---

## 🧪 Suitable Plastic Types

| Type | Source | Melting Point |
|------|--------|---------------|
| LDPE | Carry bags, packaging film | 105-115°C |
| HDPE | Milk covers, bottles | 120-135°C |
| PP | Food wrappers, containers | 130-170°C |

> ⚠️ **PVC is NOT suitable** due to toxic emissions when heated.

---

## 🛠️ Technology Stack

| Layer | Technologies |
|-------|-------------|
| **Backend** | Python, Pandas, NumPy, Scikit-learn |
| **Frontend** | Streamlit (dark theme UI) |
| **Database** | MongoDB Atlas |
| **ML Models** | SVM (RBF kernel), Random Forest (200 trees) |
| **Serialization** | Joblib (.pkl) |

---

## 📁 Project Structure

```
W-To-W/
├── app.py                          # Streamlit UI (6 pages)
├── requirements.txt                # Python dependencies
├── .streamlit/
│   └── config.toml                 # Streamlit theme config
├── data/
│   ├── generate_dataset.py         # Synthetic data generator
│   ├── plastic_waste.csv           # 500 rows
│   ├── bitumen_road_properties.csv # 500 rows
│   └── cost_dataset.csv            # 500 rows
├── models/                         # Saved .pkl model files
│   ├── svm_model.pkl
│   ├── rf_model.pkl
│   └── scaler.pkl
├── results/                        # Auto-saved chart images (PNG)
└── src/
    ├── __init__.py
    ├── config.py                   # Paths, hyperparams, columns
    ├── database.py                 # MongoDB Atlas connection & CRUD
    ├── preprocessing.py            # Load → clean → merge → scale → split
    ├── models.py                   # Train & evaluate SVM / Random Forest
    └── predict.py                  # Inference with derived labels
```

---

## ⚡ Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/PDReddyDhanu/Waste-To-Wealth.git
cd Waste-To-Wealth

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate datasets
python data/generate_dataset.py

# 4. Run the app
python -m streamlit run app.py
```

---

## 🖥️ App Pages

### 🏠 Home
Project overview with benefit cards and a one-click dataset generation button.

### 📊 Dataset Explorer
Three tabs (Plastic Waste, Bitumen Properties, Cost Analysis) with interactive dataframes, summary metrics, and auto-generated trend-line charts.

### 🤖 Model Training
Select SVM and/or Random Forest → Train → View per-target R², MAE, RMSE metrics and side-by-side comparison bar charts. Results are saved to MongoDB and charts to `results/`.

### 🔮 Prediction
Slider for plastic %, dropdown for plastic type, model selector → Get instant predictions for durability, strength, cost reduction, and a recommendation label. Results are persisted to MongoDB.

### 📜 History
Database-backed history page showing all past training runs, all past predictions, and a gallery of saved chart images — all pulled from MongoDB Atlas.

### ℹ️ About
Plastic types info, the dry-mix process, ML model descriptions, and technology stack.

---

## 📈 Model Performance

| Model | Overall R² | Status |
|-------|-----------|--------|
| SVM (RBF) | ~0.91 | ✅ Exceeds 85% target |
| Random Forest | ~0.90 | ✅ Exceeds 85% target |

---

## 🗄️ MongoDB Collections

| Collection | What's Stored |
|---|---|
| `training_runs` | Model name, R², MAE, RMSE, training time, timestamp |
| `predictions` | Plastic %, type, model used, all result metrics, timestamp |
| `datasets` | Dataset name, row/column counts, column names, timestamp |

---

## 🔄 Workflow

```
Data Collection → Data Cleaning & Preprocessing → Model Training (SVM / RF)
    → Model Evaluation → Model Storage (.pkl) → Prediction Interface
    → Text Output Display + MongoDB Storage + Chart Saving
```

---

## 📜 License

This project is for academic and research purposes.
