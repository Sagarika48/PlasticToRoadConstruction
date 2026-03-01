# ♻️ Waste to Wealth

**Plastic Waste Utilization in Road Construction using Machine Learning**

An ML-powered prediction system that determines the optimal percentage of plastic waste to mix with bitumen in road construction — reducing cost, improving durability, increasing strength, and promoting sustainable waste management.

---

## 🚀 Features

- **ML Models** — SVM & Random Forest regressors predicting 4 road performance metrics simultaneously
- **Interactive UI** — Streamlit app with 5 pages: Home, Dataset Explorer, Model Training, Prediction, About
- **Dataset Generation** — Synthetic data with domain-accurate correlations (stability peaks at 6-8% plastic)
- **Model Comparison** — Side-by-side R², MAE, RMSE comparison charts
- **Premium Design** — Dark theme with gradient cards, trend-line charts, and micro-animations

## 📊 Predicted Metrics

| Metric | Description |
|--------|-------------|
| Marshall Stability (kN) | Road strength indicator |
| Tensile Strength (MPa) | Load bearing capacity |
| Durability Score (1-100) | Expected lifespan |
| Cost Reduction (%) | Savings vs traditional bitumen |

## 🧪 Suitable Plastic Types

| Type | Source |
|------|--------|
| LDPE | Carry bags, packaging film |
| HDPE | Milk covers, bottles |
| PP | Food wrappers, containers |

> ⚠️ PVC is **not** suitable due to toxic emissions when heated.

## 🛠️ Tech Stack

- **Backend:** Python, Pandas, NumPy, Scikit-learn
- **Frontend:** Streamlit
- **Models:** SVM (RBF kernel), Random Forest (200 trees)
- **Serialization:** Joblib (.pkl)

## 📁 Project Structure

```
W-To-W/
├── app.py                          # Streamlit UI
├── requirements.txt
├── data/
│   ├── generate_dataset.py         # Synthetic data generator
│   ├── plastic_waste.csv
│   ├── bitumen_road_properties.csv
│   └── cost_dataset.csv
├── models/                         # Saved .pkl files (gitignored)
└── src/
    ├── config.py                   # Central configuration
    ├── preprocessing.py            # Data pipeline
    ├── models.py                   # Train & evaluate ML models
    └── predict.py                  # Inference module
```

## ⚡ Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate datasets
python data/generate_dataset.py

# Run the app
streamlit run app.py
```

## 📈 Model Performance

| Model | Overall R² |
|-------|-----------|
| SVM | ~0.91 |
| Random Forest | ~0.90 |

Both models exceed the **85% accuracy** target.

## 🎯 Usage

1. Open the app → Generate datasets from Home page
2. Go to **Model Training** → Train SVM and/or Random Forest
3. Go to **Prediction** → Set plastic %, select type → Get results
4. View **Dataset Explorer** for charts and data analysis

## 📜 License

This project is for academic and research purposes.
