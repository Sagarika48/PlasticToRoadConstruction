# ♻️ Waste to Wealth

**Plastic Waste Utilization in Road Construction using Machine Learning**

An ML-powered prediction system that determines the optimal percentage of plastic waste to mix with bitumen in road construction — reducing cost, improving durability, increasing strength, and promoting sustainable waste management.

---

## 🎯 What Is This Project?

Traditional road construction relies heavily on expensive bitumen. Meanwhile, plastic waste pollution is growing rapidly. This project bridges both problems by using **Machine Learning** to predict the optimal plastic-bitumen mix ratio for building durable, cost-effective, and eco-friendly roads.

```mermaid
graph LR
    A["🗑️ Plastic Waste<br/>(LDPE, HDPE, PP)"] --> B["🔬 Shredding &<br/>Processing"]
    B --> C["🔥 Mix with Hot<br/>Aggregates (160-170°C)"]
    C --> D["🛢️ Add Bitumen<br/>to Coated Aggregates"]
    D --> E["🛣️ Lay Road Using<br/>Conventional Methods"]
    
    style A fill:#e74c3c,stroke:#c0392b,color:#fff
    style B fill:#f39c12,stroke:#e67e22,color:#fff
    style C fill:#e67e22,stroke:#d35400,color:#fff
    style D fill:#3498db,stroke:#2980b9,color:#fff
    style E fill:#2ecc71,stroke:#27ae60,color:#fff
```

### The Problem ML Solves

```mermaid
graph TD
    subgraph Traditional["❌ Traditional Approach"]
        T1["Manual Lab Testing"] --> T2["Expensive & Time-Consuming"]
        T2 --> T3["Limited Data Points"]
        T3 --> T4["Suboptimal Mix Ratios"]
    end

    subgraph MLApproach["✅ ML-Based Approach"]
        M1["Train on 500+ Data Points"] --> M2["SVM & Random Forest Models"]
        M2 --> M3["Instant Predictions"]
        M3 --> M4["Optimal Mix Ratios"]
    end

    style Traditional fill:#1a1a2e,stroke:#e74c3c,color:#e8eaf6
    style MLApproach fill:#1a1a2e,stroke:#2ecc71,color:#e8eaf6
    style T4 fill:#e74c3c,stroke:#c0392b,color:#fff
    style M4 fill:#2ecc71,stroke:#27ae60,color:#fff
```

---

## 🚀 Features

- **ML Models** — SVM & Random Forest regressors predicting 4 road metrics simultaneously
- **Interactive UI** — Streamlit app with 6 pages: Home, Dataset Explorer, Model Training, Prediction, History, About
- **Dataset Generation** — Synthetic data with domain-accurate correlations (stability peaks at 6-8% plastic)
- **Model Comparison** — Side-by-side R², MAE, RMSE comparison charts
- **MongoDB Atlas** — All training runs and predictions persisted to cloud database
- **Chart Saving** — All generated charts auto-saved as PNG images to `results/`
- **Terminal Logging** — Every UI action mirrored to terminal with structured output
- **Premium Dark Theme** — Gradient cards, trend-line charts, hover animations

---

## 🏗️ System Architecture

```mermaid
graph TB
    subgraph Frontend["🖥️ Frontend — Streamlit"]
        UI1["🏠 Home"]
        UI2["📊 Dataset Explorer"]
        UI3["🤖 Model Training"]
        UI4["🔮 Prediction"]
        UI5["📜 History"]
        UI6["ℹ️ About"]
    end

    subgraph Backend["⚙️ Backend — Python"]
        PP["preprocessing.py<br/>Load → Clean → Merge<br/>→ Scale → Split"]
        ML["models.py<br/>SVM & Random Forest<br/>MultiOutputRegressor"]
        PR["predict.py<br/>Inference Engine<br/>+ Derived Labels"]
        CF["config.py<br/>Paths, Hyperparams<br/>Column Definitions"]
    end

    subgraph Storage["💾 Storage"]
        CSV["📄 CSV Datasets<br/>3 files × 500 rows"]
        PKL["📦 Model Files<br/>.pkl (Joblib)"]
        PNG["📊 Chart Images<br/>results/*.png"]
        MDB["🍃 MongoDB Atlas<br/>training_runs<br/>predictions<br/>datasets"]
    end

    subgraph DataGen["📊 Data Generation"]
        DG["generate_dataset.py<br/>Domain-Accurate<br/>Synthetic Data"]
    end

    UI2 --> PP
    UI3 --> PP
    PP --> ML
    UI4 --> PR
    PR --> ML
    UI5 --> MDB
    DG --> CSV
    ML --> PKL
    ML --> MDB
    PR --> MDB
    UI2 --> PNG
    UI3 --> PNG

    style Frontend fill:#1e2130,stroke:#3366ff,color:#e8eaf6
    style Backend fill:#1e2130,stroke:#00d68f,color:#e8eaf6
    style Storage fill:#1e2130,stroke:#f5a623,color:#e8eaf6
    style DataGen fill:#1e2130,stroke:#9b59b6,color:#e8eaf6
```

---

## 🔄 Complete Workflow

```mermaid
flowchart TD
    Start(["🚀 Start Application"]) --> GenCheck{Datasets Exist?}
    
    GenCheck -->|No| GenData["📊 Generate Synthetic Datasets<br/><i>500 rows × 3 CSVs</i>"]
    GenData --> SaveCSV["💾 Save to CSV Files"]
    SaveCSV --> SaveDB1["🍃 Save Dataset Info → MongoDB"]
    SaveDB1 --> GenCheck
    
    GenCheck -->|Yes| UserChoice{User Selects Page}
    
    UserChoice -->|Dataset Explorer| Explore["📊 View Data Tables & Charts"]
    Explore --> SaveChart1["📊 Auto-Save Charts → results/"]
    
    UserChoice -->|Model Training| SelectModel["🤖 Select SVM / Random Forest"]
    SelectModel --> Preprocess["⚙️ Preprocess Data<br/><i>Clean → Encode → Scale → Split</i>"]
    Preprocess --> Train["🏋️ Train Model<br/><i>MultiOutputRegressor</i>"]
    Train --> Evaluate["📈 Evaluate<br/><i>R², MAE, RMSE per target</i>"]
    Evaluate --> SavePKL["💾 Save Model → .pkl"]
    Evaluate --> SaveDB2["🍃 Save Metrics → MongoDB"]
    Evaluate --> SaveChart2["📊 Save Comparison Charts → results/"]
    Evaluate --> ShowMetrics["📋 Display Metrics in UI"]
    Evaluate --> LogTerminal1["🖥️ Log Metrics to Terminal"]
    
    UserChoice -->|Prediction| InputParams["🔮 Input: Plastic %, Type, Model"]
    InputParams --> LoadModel["📦 Load Trained Model + Scaler"]
    LoadModel --> Predict["🧮 Run Prediction"]
    Predict --> DeriveLabels["🏷️ Derive Labels<br/><i>Durability: High/Med/Low</i><br/><i>Recommendation: ✅/⚠️/❌</i>"]
    DeriveLabels --> ShowResults["📋 Display Results in UI"]
    DeriveLabels --> SaveDB3["🍃 Save Prediction → MongoDB"]
    DeriveLabels --> LogTerminal2["🖥️ Log Results to Terminal"]
    
    UserChoice -->|History| FetchDB["🍃 Fetch from MongoDB"]
    FetchDB --> ShowHistory["📜 Show Training Runs +<br/>Predictions + Chart Gallery"]
    
    style Start fill:#00d68f,stroke:#00a86b,color:#fff
    style GenData fill:#9b59b6,stroke:#8e44ad,color:#fff
    style Train fill:#3366ff,stroke:#2952cc,color:#fff
    style Predict fill:#f5a623,stroke:#d4941e,color:#fff
    style SaveDB1 fill:#2ecc71,stroke:#27ae60,color:#fff
    style SaveDB2 fill:#2ecc71,stroke:#27ae60,color:#fff
    style SaveDB3 fill:#2ecc71,stroke:#27ae60,color:#fff
```

---

## 📊 Data Pipeline

```mermaid
flowchart LR
    subgraph Input["📥 Raw Datasets"]
        D1["plastic_waste.csv<br/><i>type, availability,<br/>size, melting point</i>"]
        D2["bitumen_road_properties.csv<br/><i>plastic%, stability,<br/>strength, durability</i>"]
        D3["cost_dataset.csv<br/><i>bitumen cost, processing<br/>cost, maintenance cost</i>"]
    end

    subgraph Preprocess["⚙️ Preprocessing"]
        P1["Load CSVs"]
        P2["Merge by plastic_%"]
        P3["Handle Missing Values<br/><i>Median imputation</i>"]
        P4["Encode Plastic Type<br/><i>LDPE→0, HDPE→1, PP→2</i>"]
        P5["Feature Scaling<br/><i>StandardScaler</i>"]
        P6["Train/Test Split<br/><i>80/20 ratio</i>"]
    end

    subgraph Output["📤 Ready for ML"]
        O1["X_train (400×5)"]
        O2["X_test (100×5)"]
        O3["y_train (400×4)"]
        O4["y_test (100×4)"]
    end

    D1 --> P1
    D2 --> P1
    D3 --> P1
    P1 --> P2 --> P3 --> P4 --> P5 --> P6
    P6 --> O1
    P6 --> O2
    P6 --> O3
    P6 --> O4

    style Input fill:#1e2130,stroke:#9b59b6,color:#e8eaf6
    style Preprocess fill:#1e2130,stroke:#3366ff,color:#e8eaf6
    style Output fill:#1e2130,stroke:#00d68f,color:#e8eaf6
```

### Feature & Target Columns

```mermaid
graph LR
    subgraph Features["📥 Input Features (5)"]
        F1["plastic_pct"]
        F2["bitumen_pct"]
        F3["plastic_type_encoded"]
        F4["softening_point"]
        F5["penetration_value"]
    end

    ML["🤖 ML Model<br/>(MultiOutputRegressor)"]

    subgraph Targets["📤 Predicted Targets (4)"]
        T1["Marshall Stability (kN)"]
        T2["Tensile Strength (MPa)"]
        T3["Durability Score (1-100)"]
        T4["Cost Reduction (%)"]
    end

    F1 --> ML
    F2 --> ML
    F3 --> ML
    F4 --> ML
    F5 --> ML
    ML --> T1
    ML --> T2
    ML --> T3
    ML --> T4

    style Features fill:#1e2130,stroke:#3366ff,color:#e8eaf6
    style ML fill:#00d68f,stroke:#00a86b,color:#fff
    style Targets fill:#1e2130,stroke:#f5a623,color:#e8eaf6
```

---

## 🤖 ML Model Comparison

```mermaid
graph TD
    subgraph SVM["Support Vector Machine"]
        S1["Kernel: RBF"]
        S2["C: 100, Gamma: scale"]
        S3["Epsilon: 0.1"]
        S4["Wrapped in MultiOutputRegressor"]
        S5["✅ R² ≈ 0.91"]
    end

    subgraph RF["Random Forest"]
        R1["Trees: 200"]
        R2["Max Depth: 12"]
        R3["Min Samples Split: 5"]
        R4["Wrapped in MultiOutputRegressor"]
        R5["✅ R² ≈ 0.90"]
    end

    Data["📊 Training Data<br/>(400 samples × 5 features)"] --> SVM
    Data --> RF

    SVM --> Eval["📈 Evaluation<br/>R², MAE, RMSE"]
    RF --> Eval
    Eval --> Best["🏆 Best Model Selected<br/>for Predictions"]

    style SVM fill:#1e2130,stroke:#3366ff,color:#e8eaf6
    style RF fill:#1e2130,stroke:#00d68f,color:#e8eaf6
    style Data fill:#9b59b6,stroke:#8e44ad,color:#fff
    style Best fill:#f5a623,stroke:#d4941e,color:#fff
```

---

## 🔮 Prediction Flow

```mermaid
flowchart LR
    Input["👤 User Input<br/>Plastic: 6%<br/>Type: LDPE<br/>Model: RF"] 
    --> Scale["⚖️ Scale Features<br/>(StandardScaler)"]
    --> Model["🤖 Loaded Model<br/>(.pkl file)"]
    --> Raw["📊 Raw Predictions<br/>stability, strength,<br/>durability, cost"]
    --> Labels["🏷️ Derive Labels"]
    --> Output["📋 Final Output"]

    subgraph OutputDetails["Example Output"]
        O1["Stability: 11.95 kN"]
        O2["Strength: 3.28 MPa (+64%)"]
        O3["Durability: High (75)"]
        O4["Cost Reduction: 17.5%"]
        O5["✅ Highly Recommended"]
    end

    Output --> OutputDetails

    style Input fill:#3366ff,stroke:#2952cc,color:#fff
    style Model fill:#00d68f,stroke:#00a86b,color:#fff
    style Output fill:#f5a623,stroke:#d4941e,color:#fff
    style OutputDetails fill:#1e2130,stroke:#00d68f,color:#e8eaf6
```

---

## 📊 Domain Relationships

These are the key domain rules encoded in the synthetic dataset:

```mermaid
graph TD
    PP["Plastic Percentage (0-15%)"]
    
    PP -->|"Peaks at 6-8%"| MS["📈 Marshall Stability"]
    PP -->|"Increases linearly"| SP["📈 Softening Point"]
    PP -->|"Decreases linearly"| PV["📉 Penetration Value"]
    PP -->|"Peaks at ~8%"| TS["📈 Tensile Strength"]
    PP -->|"Improves up to ~10%"| DS["📈 Durability Score"]
    PP -->|"Reduces bitumen cost"| BC["📉 Bitumen Cost"]
    PP -->|"Increases processing"| PC["📈 Processing Cost"]
    PP -->|"Net reduction grows"| CR["📈 Cost Reduction %"]

    style PP fill:#9b59b6,stroke:#8e44ad,color:#fff
    style MS fill:#00d68f,stroke:#00a86b,color:#fff
    style TS fill:#00d68f,stroke:#00a86b,color:#fff
    style DS fill:#00d68f,stroke:#00a86b,color:#fff
    style CR fill:#00d68f,stroke:#00a86b,color:#fff
    style SP fill:#3366ff,stroke:#2952cc,color:#fff
    style PV fill:#e74c3c,stroke:#c0392b,color:#fff
    style BC fill:#e74c3c,stroke:#c0392b,color:#fff
    style PC fill:#f5a623,stroke:#d4941e,color:#fff
```

---

## 📊 Predicted Metrics

| Metric                   | Description                    |
| ------------------------ | ------------------------------ |
| Marshall Stability (kN)  | Road strength indicator        |
| Tensile Strength (MPa)   | Load bearing capacity          |
| Durability Score (1-100) | Expected lifespan              |
| Cost Reduction (%)       | Savings vs traditional bitumen |

---

## 🧪 Suitable Plastic Types

| Type | Source                     | Melting Point |
| ---- | -------------------------- | ------------- |
| LDPE | Carry bags, packaging film | 105-115°C     |
| HDPE | Milk covers, bottles       | 120-135°C     |
| PP   | Food wrappers, containers  | 130-170°C     |

> ⚠️ **PVC is NOT suitable** due to toxic emissions when heated.

---

## 🛠️ Technology Stack

| Layer               | Technologies                                |
| ------------------- | ------------------------------------------- |
| **Backend**         | Python, Pandas, NumPy, Scikit-learn         |
| **Frontend**        | Streamlit (dark theme UI)                   |
| **Database**        | MongoDB Atlas                               |
| **ML Models**       | SVM (RBF kernel), Random Forest (200 trees) |
| **Serialization**   | Joblib (.pkl)                               |

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

| Model         | Overall R² | Status                |
| ------------- | ---------- | --------------------- |
| SVM (RBF)     | ~0.91      | ✅ Exceeds 85% target |
| Random Forest | ~0.90      | ✅ Exceeds 85% target |

---

## 🗄️ MongoDB Collections

```mermaid
erDiagram
    training_runs {
        datetime timestamp
        string model_name
        float train_time_seconds
        float overall_r2
        float overall_mae
        float overall_rmse
        object per_target_metrics
    }

    predictions {
        datetime timestamp
        float plastic_pct
        string plastic_type
        string model_used
        float marshall_stability
        float tensile_strength
        float durability_score
        string durability_label
        float cost_reduction_pct
        float strength_improvement
        string recommendation
    }

    datasets {
        datetime timestamp
        string dataset_name
        int rows
        int columns
        array column_names
    }
```

---

## 📜 License

This project is for academic and research purposes.
