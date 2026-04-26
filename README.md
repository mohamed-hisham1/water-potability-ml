# 💧 Water Potability Prediction — ML Project

An end-to-end machine learning project that predicts whether a water sample is **safe for drinking (potable)** based on its physical and chemical properties. The project includes data preprocessing, exploratory data analysis, model training, and an interactive **Streamlit web application** for real-time predictions.

---

## 🗂️ Project Structure

```
water-potability-ml/
│
├── 01_preprocessing.ipynb       # Data cleaning & EDA
├── 02_modelling.ipynb           # Model training & evaluation
├── water_quality_analysis.ipynb # Additional analysis notebook
├── app.py                       # Streamlit web application
├── potability.csv               # Dataset
├── best_model.pkl               # Best saved model artifact
├── water_quality_model.pkl      # Trained Random Forest model
├── scaler.pkl                   # Feature scaler artifact
├── requirements.txt             # Python dependencies
└── .gitignore
```

---

## 📊 Dataset

The dataset (`potability.csv`) contains water quality measurements for **3,276 water bodies**. Each row represents a water sample described by 9 features and a binary target variable.

| Feature | Unit | Description |
|---|---|---|
| `ph` | pH units | Acidity/alkalinity of water (WHO standard: 6.5–8.5) |
| `Hardness` | mg/L | Caused by dissolved calcium and magnesium salts |
| `Solids` | ppm | Total dissolved solids (TDS) |
| `Chloramines` | ppm | Disinfectant used in water treatment |
| `Sulfate` | mg/L | Naturally occurring dissolved minerals |
| `Conductivity` | μS/cm | Electrical conductivity of the water |
| `Organic_carbon` | ppm | Amount of carbon from organic compounds |
| `Trihalomethanes` | μg/L | Chemicals produced during chlorination |
| `Turbidity` | NTU | Measure of water clarity |
| `Potability` | — | **Target**: 1 = Potable, 0 = Not Potable |

**Class distribution:** ~61% not potable, ~39% potable (imbalanced dataset).

---

## 🔬 Notebooks

### `01_preprocessing.ipynb`
Covers the full data preparation pipeline:
- Loading and inspecting the dataset
- Detecting and handling missing values (median imputation)
- Exploratory Data Analysis (EDA) with distributions and correlation heatmaps
- Feature scaling and preparation for modelling

### `02_modelling.ipynb`
Covers model development and evaluation:
- Train/test splitting (80/20, stratified)
- Training a **Random Forest Classifier** (200 estimators)
- Evaluation with accuracy, ROC-AUC, confusion matrix, and classification report
- Feature importance analysis
- Saving the best model to `water_quality_model.pkl`

### `water_quality_analysis.ipynb`
Supplementary analysis notebook with additional visualisations and quality checks.

---

## 🤖 Model

| Detail | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Estimators | 200 trees |
| Train / Test split | 80% / 20% (stratified) |
| Missing value strategy | Median imputation |
| Saved artifact | `water_quality_model.pkl` |

---

## 🖥️ Streamlit Web Application (`app.py`)

The app provides an interactive, dark-themed dashboard with four pages:

### 📊 Data Overview
- Dataset statistics (sample count, class balance)
- Donut chart showing the potability split
- Interactive feature explorer with per-class KDE plots
- Full 3×3 KDE grid for all 9 features
- Statistical summary table

### 🤖 Model Training
- One-click model training in the browser
- Displays test accuracy, ROC-AUC, and sample counts
- Confusion matrix, ROC curve, and classification report visualisations
- Saves the trained model automatically

### 🔮 Make Prediction
- 9 interactive sliders (one per feature, with real-world units and defaults)
- Instant prediction result card (✅ Potable / ❌ Not Potable)
- Confidence percentage and probability breakdown bar chart

### 📈 Model Analysis
- Feature importance lollipop chart with cumulative importance line
- Full correlation heatmap (triangular, coolwarm palette)
- Ranked feature importance table with Gini scores

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

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

> **Note:** Make sure `potability.csv` is present in the same directory as `app.py` before launching.

---

## 📦 Dependencies

```
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

Install all at once with:

```bash
pip install -r requirements.txt
```

---

## ☁️ Deployment

This project is configured for deployment on **Hugging Face Spaces** using the Streamlit SDK (v1.28.0). The `README.md` front matter specifies:

```yaml
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
```

---

## 🧠 How It Works

1. **Load data** → `potability.csv` is read and missing values are imputed with column medians.
2. **Train** → A `RandomForestClassifier` is fit on 80% of the data.
3. **Evaluate** → Accuracy, AUC-ROC, and a confusion matrix are computed on the held-out 20%.
4. **Predict** → User-provided slider values are passed through the trained model, returning a potability label and confidence score.
5. **Persist** → The model is serialised with `joblib` and loaded on subsequent app sessions.

---

## 📄 License

This project is open-source. Feel free to fork, modify, and use it for your own learning or research.

---

*Built with Python · scikit-learn · Streamlit · Matplotlib · Seaborn*
