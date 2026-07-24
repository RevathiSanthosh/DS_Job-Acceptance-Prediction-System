# 🎯 Job Acceptance Prediction System
**GUVI HCL · Fullstack Data Science with Generative & Agentic AI Program**

---

## 📋 Project Overview
| Field | Details |
|-------|---------|
| Domain | HR Analytics & Predictive Modeling |
| Dataset | 50,000 synthetic candidate records |
| Target | Predict: Placed / Not Placed |
| Best Model | Selected automatically by AUC-ROC from 6 algorithms |

---

## 🏗️ Project Structure
```
job_acceptance_prediction/
├── setup.py                    ← ONE-CLICK pipeline runner
├── requirements.txt
├── README.md
├── config/settings.py          ← Paths, colours, constants
├── data/generator.py           ← 50K synthetic HR dataset
├── preprocessing/pipeline.py  ← 6-step cleaning pipeline
├── features/engineering.py     ← Derived features
├── eda/visualizations.py       ← 15 EDA charts
├── models/training.py          ← 6 ML models + best selection
├── dashboard/app.py            ← Streamlit BI dashboard
└── outputs/
    ├── eda_charts/             ← 20 PNG charts
    ├── model_plots/            ← Saved model .pkl files
    └── reports/                ← Cleaning + model reports
```

---

## ⚡ Quick Start (3 Steps)

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run full pipeline
```bash
python setup.py
```
This generates data → cleans → engineers features → 15 EDA charts → trains 6 models → selects best

### Step 3 — Launch dashboard
```bash
streamlit run dashboard/app.py
```
Open http://localhost:8501 in browser

---

## 🤖 ML Algorithms Compared

| # | Algorithm | Type |
|---|-----------|------|
| 1 | Logistic Regression | Linear |
| 2 | K-Nearest Neighbors | Instance-based |
| 3 | Decision Tree | Tree-based |
| 4 | Random Forest | Ensemble (Bagging) |
| 5 | Gradient Boosting | Ensemble (Boosting) |

**Selection Criterion:** Highest AUC-ROC score on held-out test set (20%)

**Evaluation Metrics:** Accuracy · Precision · Recall · F1 · AUC-ROC · 5-Fold CV

---

## 📊 Dashboard Pages
| Page | Content |
|------|---------|
| 📊 Overview | 7 KPIs + placement trends + filters |
| 📈 EDA Charts | Academic, Interview, Skills, Correlation tabs |
| 🤖 Model Results | ROC curves, comparison, feature importance |
| 🔮 Predict | Real-time candidate acceptance prediction |

---

## 🔧 Data Cleaning Steps
1. Remove duplicates (semantic + exact)
2. Standardize inconsistent categoricals (tier1→Tier 1, etc.)
3. Impute missing values (median for numeric, mode for categorical)
4. Logical consistency checks (clip outliers)
5. Encode target variable (Placed=1, Not Placed=0)
6. Label-encode categoricals + StandardScaler for numerics

---

*Subhash Govindharaj · Shadiya P.P · Nehlath Harmain H.E*
