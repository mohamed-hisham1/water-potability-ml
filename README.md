# 💧 Water Potability Prediction — ML Project

An end-to-end machine learning project that predicts whether a water sample is **safe for drinking (potable)** based on its physical and chemical properties. The project is structured across three Jupyter notebooks and a full-featured **Streamlit web application**.

---

## 🗂️ Project Structure

```
water-potability-ml/
│
├── 01_preprocessing.ipynb        # Data cleaning, EDA, feature engineering & scaling
├── 02_modelling.ipynb            # Multi-model training, tuning & selection
├── water_quality_analysis.ipynb  # Standalone analysis notebook mirroring app.py
├── app.py                        # Streamlit web application (786 lines)
│
├── potability.csv                # Raw dataset (3,276 samples × 10 columns)
├── X_train.csv / X_test.csv      # Train/test feature splits (saved by notebook 1)
├── X_train_scaled.csv / X_test_scaled.csv  # StandardScaler output
├── y_train.csv / y_test.csv      # Label splits
│
├── scaler.pkl                    # Fitted StandardScaler (saved by notebook 1)
├── best_model.pkl                # Best model chosen in notebook 2
├── water_quality_model.pkl       # Random Forest model (used by app.py)
│
├── requirements.txt              # Python dependencies
└── .gitignore
```

---

## 📊 Dataset

`potability.csv` contains water quality measurements for **3,276 water bodies**. Three features contain missing values, which are filled with their per-column **median** (robust to outliers).

| Feature | Unit | Description |
|---|---|---|
| `ph` | pH units | Acid–base balance (WHO safe range: 6.5–8.5) |
| `Hardness` | mg/L | Dissolved calcium & magnesium salts |
| `Solids` | ppm | Total dissolved solids (TDS) |
| `Chloramines` | ppm | Disinfectant added during water treatment |
| `Sulfate` | mg/L | Naturally occurring dissolved minerals |
| `Conductivity` | μS/cm | Electrical conductivity of the water |
| `Organic_carbon` | ppm | Carbon from organic compounds |
| `Trihalomethanes` | μg/L | By-products formed during chlorination |
| `Turbidity` | NTU | Clarity / light-scattering of the water |
| `Potability` | — | **Target**: `1` = Potable, `0` = Not Potable |

**Class distribution:** ~60% Not Potable / ~40% Potable — mild imbalance addressed via stratified splits and `class_weight='balanced'` where applicable.

---

## 🔬 Notebook 1 — Preprocessing & EDA (`01_preprocessing.ipynb`)

**Prerequisite for Notebook 2.** All artefacts produced here are saved to disk and loaded by the modelling notebook.

### Steps

1. **Libraries** — numpy, pandas, matplotlib, seaborn, scikit-learn, joblib
2. **Load dataset** — inspect shape, dtypes, and `.describe()`
3. **Target variable analysis** — pie chart + annotated bar chart; imbalance ratio printed
4. **Missing value analysis** — missing count heatmap; fill strategy: **median per column**
5. **Duplicate check** — drop any duplicate rows found
6. **Univariate analysis** — 3×3 histogram grid with mean & median lines for all 9 features
7. **Outlier analysis** — 3×3 box-plot grid
8. **Bivariate analysis** — KDE plots split by potability class for each feature
9. **Correlation heatmap** — lower-triangle coolwarm heatmap; absolute correlation with target printed
10. **Feature engineering** — 7 new columns added:
    - `Solids_Conductivity` = Solids × Conductivity
    - `Chloramines_THMs` = Chloramines × Trihalomethanes
    - `Hardness_Sulfate` = Hardness × Sulfate
    - `Turbidity_Organic` = Turbidity × Organic_carbon
    - `log_Solids` = log1p(Solids)
    - `log_Conductivity` = log1p(Conductivity)
    - `ph_squared` = ph²
11. **Train/Test split** — 80/20, stratified, `random_state=42`
12. **Scaling** — `StandardScaler` fitted on train, applied to both splits
13. **Save artefacts** — `X_train.csv`, `X_test.csv`, `X_train_scaled.csv`, `X_test_scaled.csv`, `y_train.csv`, `y_test.csv`, `scaler.pkl`

---

## 🤖 Notebook 2 — Modelling (`02_modelling.ipynb`)

**Prerequisite:** Run Notebook 1 first. Loads all CSV artefacts and `scaler.pkl`.

### Models Trained

| Model | Library | Baseline estimators |
|---|---|---|
| Random Forest | scikit-learn | 100 trees |
| XGBoost | xgboost | default |
| LightGBM | lightgbm | default |

### Workflow per Model

Each of the three models goes through the same two-stage pipeline:

**Stage 1 — Baseline:** train with default/minimal hyperparameters, compute accuracy, F1, and ROC-AUC.

**Stage 2 — Hyperparameter Tuning:** `RandomizedSearchCV` with 30 iterations (`N_ITER=30`), 5-fold `StratifiedKFold`, scoring on **F1**.

Search spaces used:

- **Random Forest:** `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `class_weight`, `bootstrap`
- **XGBoost:** `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `gamma`, `min_child_weight`, `reg_alpha`, `reg_lambda`, `scale_pos_weight`
- **LightGBM:** `n_estimators`, `max_depth`, `learning_rate`, `num_leaves`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `class_weight`

### Evaluation & Visualisations

| Step | Output |
|---|---|
| Full metrics table | Accuracy, Precision, Recall, F1, ROC-AUC, CV-F1 Mean, CV-F1 Std, Train Time — all 6 models |
| Metric bar chart | Grouped bars for all 5 metrics across the 3 tuned models |
| Confusion matrices | One `ConfusionMatrixDisplay` per tuned model (3 side-by-side) |
| Classification reports | Per-class precision/recall/F1 printed for all 3 tuned models |
| ROC curves | All 3 tuned models on one axes with AUC values in the legend |
| Cross-validation F1 | Per-fold bar charts (5 folds × 3 models) with mean line and shaded ±1 std band |
| Feature importance | Top-15 horizontal bar charts per model + normalised comparison grouped bar chart |
| Baseline vs Tuned gain | Paired bar chart with annotated F1 gain (+Δ) per model |

### Model Selection

A **composite score** (0.5 × F1 + 0.5 × ROC-AUC) is computed for the three tuned models. The highest-scoring model is selected automatically and saved as `best_model.pkl`.

---

## 📓 Standalone Analysis Notebook (`water_quality_analysis.ipynb`)

A self-contained notebook that **mirrors and extends** the logic of `app.py`. It loads `potability.csv` directly (no notebook-1 artefacts needed) and applies the same median-fill strategy.

| Section | Content |
|---|---|
| 1 | Libraries & config — same constants as `app.py` (`N_ESTIMATORS=200`, `TEST_SIZE=0.2`, `random_state=42`) |
| 2 | Data loading & cleaning — missing-value summary table + heatmap |
| 3 | Dataset overview — sample counts, class balance printed |
| 4 | EDA — target distribution pie/bar, 3×3 histogram grid, 3×3 KDE-by-class grid |
| 5 | Correlation heatmap + absolute correlation with target as styled DataFrame |
| 6 | Model training — `RandomForestClassifier(n_estimators=200)`, same config as `app.py` |
| 7 | Evaluation — metrics table, classification report, confusion matrix + ROC side-by-side, 5-fold CV bar chart |
| 8 | Feature importance — horizontal bar + Pareto chart with cumulative % secondary axis |
| 9 | Prediction interface — single-sample prediction using app.py slider defaults; batch prediction on test set with misclassified-row highlight |
| 10 | Model persistence — `joblib` save/load with round-trip assertion check |

---

## 🖥️ Streamlit App (`app.py`)

A dark-themed, 786-line Streamlit dashboard with four pages accessible from the sidebar.

### Design System

Custom CSS injects three Google Fonts (Syne, DM Sans, DM Mono) and defines a GitHub-dark colour palette:

| Token | Hex | Usage |
|---|---|---|
| `DARK_BG` | `#0d1117` | Page & figure background |
| `CARD_BG` | `#161b22` | Metric cards, axes background |
| `ACCENT_TEAL` | `#00c9a7` | Primary accent, potable result cards |
| `ACCENT_RED` | `#ff6b6b` | Not-potable results, FP/FN cells |
| `ACCENT_GOLD` | `#ffd166` | Mean lines, cumulative importance curves |
| `TEXT_DIM` | `#8b949e` | Tick labels, secondary text |

All matplotlib charts share a global theme applied once at startup via `set_mpl_theme()`.

---

### 📊 Page 1 — Data Overview

- **4 metric cards:** Total Samples, Features, Potable count (%), Not Potable count (%)
- **Target distribution:** donut chart (wedge-style pie with width=0.52) + annotated horizontal bar chart showing class counts and imbalance ratio
- **Feature explorer:** `st.selectbox` to choose any of the 9 features; renders histogram + per-class KDE overlay with mean (gold dashed) and median (purple dotted) vertical lines
- **All feature distributions:** full 3×3 KDE grid for all 9 features simultaneously with a shared legend
- **Statistical summary:** `.describe().T` styled with a Blues gradient

---

### 🤖 Page 2 — Model Training

- Trains `RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)` on an 80/20 stratified split
- Missing values filled with `df.fillna(df.median())` before training
- **4 metric cards:** Test Accuracy, ROC-AUC, Train samples, Test samples
- **Confusion matrix** — custom `FancyBboxPatch` rounded cells (teal for TP/TN, red for FP/FN) with count and % annotations; no `seaborn` heatmap
- **ROC curve** — filled area under curve + optimal threshold dot (nearest point to top-left corner) highlighted in gold
- **Classification report** — visual table chart with inline horizontal bar segments coloured per metric (teal/gold/purple)
- Model saved to `water_quality_model.pkl` via `joblib`

---

### 🔮 Page 3 — Make Prediction

- **9 sliders** arranged in a 3-column layout with real-world units and defaults matching dataset means:

| Slider | Range | Default |
|---|---|---|
| ph | 0.0 – 14.0 | 7.0 |
| Hardness | 0.0 – 500.0 mg/L | 196.0 |
| Solids | 0.0 – 60,000 ppm | 22,000 |
| Chloramines | 0.0 – 15.0 ppm | 7.0 |
| Sulfate | 0.0 – 500.0 mg/L | 333.0 |
| Conductivity | 0.0 – 800.0 μS/cm | 426.0 |
| Organic_carbon | 0.0 – 30.0 ppm | 14.0 |
| Trihalomethanes | 0.0 – 130.0 μg/L | 66.0 |
| Turbidity | 0.0 – 10.0 NTU | 4.0 |

- **Prediction result card:** colour-coded `<div>` (teal border = ✅ Potable / red border = ❌ Not Potable) with confidence percentage
- **Probability breakdown:** dual horizontal bar chart showing both class probabilities with a 0.5 decision-boundary line
- Auto-loads `water_quality_model.pkl`; if not found, offers an inline "Train Now" button that saves the model and calls `st.rerun()` to unlock the sliders

---

### 📈 Page 4 — Model Analysis

- **Feature importance:** lollipop chart with `cool` colormap (sorted ascending) plus a secondary twin-x axis showing cumulative importance %; 80% threshold marked with a gold dotted line
- **Correlation heatmap:** lower-triangle `sns.heatmap` with coolwarm palette, full annotation, and colorbar
- **Importance table:** sorted DataFrame displaying Gini Importance, Importance %, and Cumulative % with a Greens gradient

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/mohamed-hisham1/water-potability-ml.git
cd water-potability-ml
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> For Notebook 2, also install: `pip install xgboost lightgbm`

### 3. Run the notebooks in order

```bash
jupyter notebook 01_preprocessing.ipynb   # generates CSV artefacts + scaler.pkl
jupyter notebook 02_modelling.ipynb       # trains all models, saves best_model.pkl
```

### 4. Launch the Streamlit app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Ensure `potability.csv` is in the same directory.

---

## 📦 Dependencies (`requirements.txt`)

```
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

---

## ☁️ Deployment (Hugging Face Spaces)

The repository is configured for Hugging Face Spaces:

```yaml
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
```

---

## 🧠 Pipeline Overview

```
potability.csv
     │
     ▼
01_preprocessing.ipynb
  ├─ Median imputation for missing values
  ├─ EDA: distributions, outliers, KDE by class, correlation
  ├─ Feature engineering: 7 new interaction/log/polynomial features
  ├─ StandardScaler → scaler.pkl
  └─ Train/test CSV splits saved
     │
     ▼
02_modelling.ipynb
  ├─ Baseline: Random Forest · XGBoost · LightGBM
  ├─ RandomizedSearchCV (30 iter, 5-fold StratifiedKFold, F1 scoring)
  ├─ Full evaluation: accuracy, precision, recall, F1, ROC-AUC, CV-F1
  ├─ Visualisations: confusion matrices, ROC curves, feature importance, gain analysis
  └─ Composite score (F1 + AUC) → best_model.pkl
     │
     ▼
app.py (Streamlit)
  ├─ Loads potability.csv + water_quality_model.pkl
  └─ 4-page dashboard: Overview · Training · Prediction · Analysis
```

---

*Built with Python · scikit-learn · XGBoost · LightGBM · Streamlit · Matplotlib · Seaborn*
