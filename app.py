import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Arc
import matplotlib.patheffects as pe
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
import joblib

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Water Quality Prediction",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global palette (used by both Streamlit CSS and matplotlib) ─────────────
DARK_BG     = "#0d1117"
CARD_BG     = "#161b22"
ACCENT_TEAL = "#00c9a7"
ACCENT_RED  = "#ff6b6b"
ACCENT_GOLD = "#ffd166"
TEXT_MAIN   = "#e6edf3"
TEXT_DIM    = "#8b949e"
GRID_COLOR  = "#21262d"

# ── Matplotlib theme (applied once) ───────────────────────────────────────
def set_mpl_theme():
    plt.rcParams.update({
        "figure.facecolor":  DARK_BG,
        "axes.facecolor":    CARD_BG,
        "axes.edgecolor":    GRID_COLOR,
        "axes.labelcolor":   TEXT_MAIN,
        "axes.titlecolor":   TEXT_MAIN,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "axes.titlepad":     14,
        "axes.labelsize":    10,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.color":        GRID_COLOR,
        "grid.linewidth":    0.6,
        "xtick.color":       TEXT_DIM,
        "ytick.color":       TEXT_DIM,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "text.color":        TEXT_MAIN,
        "legend.facecolor":  CARD_BG,
        "legend.edgecolor":  GRID_COLOR,
        "legend.fontsize":   9,
        "figure.dpi":        120,
    })

set_mpl_theme()

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    background-color: {DARK_BG};
    color: {TEXT_MAIN};
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: {CARD_BG} !important;
    border-right: 1px solid {GRID_COLOR};
}}
[data-testid="stSidebar"] * {{ color: {TEXT_MAIN} !important; }}

/* Main header */
.app-title {{
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(90deg, {ACCENT_TEAL}, {ACCENT_GOLD});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem;
}}
.app-subtitle {{
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: {TEXT_DIM};
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 2rem;
}}

/* Section headers */
h2, h3 {{ font-family: 'Syne', sans-serif !important; font-weight: 700 !important; color: {TEXT_MAIN} !important; }}

/* Metric cards */
[data-testid="stMetric"] {{
    background: {CARD_BG};
    border: 1px solid {GRID_COLOR};
    border-radius: 12px;
    padding: 18px 22px !important;
}}
[data-testid="stMetricLabel"] {{ color: {TEXT_DIM} !important; font-size: 0.75rem !important; letter-spacing: 1px; text-transform: uppercase; }}
[data-testid="stMetricValue"] {{ color: {ACCENT_TEAL} !important; font-family: 'Syne', sans-serif !important; font-size: 2rem !important; }}

/* Buttons */
.stButton > button {{
    background: linear-gradient(135deg, {ACCENT_TEAL}, #00a896) !important;
    color: #0d1117 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 26px !important;
    letter-spacing: 0.5px;
    transition: transform 0.15s, box-shadow 0.15s;
}}
.stButton > button:hover {{ transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,201,167,0.3) !important; }}

/* Selectbox / sliders */
[data-testid="stSelectbox"] select,
.stSlider {{ accent-color: {ACCENT_TEAL}; }}

/* Prediction cards */
.card-potable {{
    background: linear-gradient(135deg, rgba(0,201,167,0.12), rgba(0,201,167,0.04));
    border: 1px solid {ACCENT_TEAL};
    border-left: 4px solid {ACCENT_TEAL};
    border-radius: 12px;
    padding: 24px 28px;
    margin: 12px 0;
}}
.card-not-potable {{
    background: linear-gradient(135deg, rgba(255,107,107,0.12), rgba(255,107,107,0.04));
    border: 1px solid {ACCENT_RED};
    border-left: 4px solid {ACCENT_RED};
    border-radius: 12px;
    padding: 24px 28px;
    margin: 12px 0;
}}
.card-title {{ font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 700; margin-bottom: 6px; }}
.card-sub   {{ font-family: 'DM Mono', monospace; font-size: 0.85rem; color: {TEXT_DIM}; }}

/* Dataframe */
[data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; border: 1px solid {GRID_COLOR}; }}

/* Success / info / warning */
.stSuccess, .stInfo, .stWarning {{ border-radius: 10px !important; }}

/* Divider */
hr {{ border-color: {GRID_COLOR} !important; }}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────
FEATURE_NAMES = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

FEATURE_UNITS = {
    'ph': 'pH units', 'Hardness': 'mg/L', 'Solids': 'ppm',
    'Chloramines': 'ppm', 'Sulfate': 'mg/L', 'Conductivity': 'μS/cm',
    'Organic_carbon': 'ppm', 'Trihalomethanes': 'μg/L', 'Turbidity': 'NTU'
}


# ══════════════════════════════════════════════════════════════════════════
# Model class
# ══════════════════════════════════════════════════════════════════════════
class WaterQualityPredictor:
    def __init__(self):
        self.model = None
        self.feature_names = FEATURE_NAMES

    def load_data(self):
        try:
            df = pd.read_csv('potability.csv')
            # FIX 1: Safe and efficient imputation for Pandas 2.0+
            df = df.fillna(df.median())
            return df
        except FileNotFoundError:
            st.error("❌ 'potability.csv' not found in the working directory.")
            return None

    def train_model(self, df):
        X = df[self.feature_names]
        y = df['Potability']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        self.model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        return X_train, X_test, y_train, y_test, y_pred, y_prob

    def predict_water_quality(self, input_features):
        if self.model is None:
            return None
        input_df = pd.DataFrame([input_features], columns=self.feature_names)
        return self.model.predict(input_df)[0], self.model.predict_proba(input_df)[0]

    def get_feature_importance(self):
        if self.model is None:
            return None
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════
# Visualisation helpers
# ══════════════════════════════════════════════════════════════════════════

def gradient_bar(ax, x, heights, color_start, color_end, **kwargs):
    """Draw bars with a vertical gradient via a stacked approach."""
    for xi, h in zip(x, heights):
        n = 100
        for j in range(n):
            frac  = j / n
            alpha = 0.4 + 0.6 * frac
            c_r = int(int(color_start[1:3], 16) * (1-frac) + int(color_end[1:3], 16) * frac)
            c_g = int(int(color_start[3:5], 16) * (1-frac) + int(color_end[3:5], 16) * frac)
            c_b = int(int(color_start[5:7], 16) * (1-frac) + int(color_end[5:7], 16) * frac)
            color = f"#{c_r:02x}{c_g:02x}{c_b:02x}"
            ax.bar(xi, h / n, bottom=h * j / n, width=0.6, color=color, alpha=alpha, **kwargs)


def fig_target_distribution(df):
    """Donut + annotated bar side-by-side."""
    counts = df['Potability'].value_counts().sort_index()
    labels = ['Not Potable', 'Potable']
    colors = [ACCENT_RED, ACCENT_TEAL]

    fig = plt.figure(figsize=(13, 5), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # ── Donut ──────────────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    ax0.set_facecolor(DARK_BG)
    wedges, texts, autotexts = ax0.pie(
        counts, labels=None, autopct='%1.1f%%',
        colors=colors, startangle=90,
        pctdistance=0.72,
        wedgeprops={"width": 0.52, "edgecolor": DARK_BG, "linewidth": 3}
    )
    for at in autotexts:
        at.set(fontsize=13, fontweight='bold', color=DARK_BG,
               fontfamily='DM Sans')

    # Centre label
    ax0.text(0, 0.08, f"{len(df):,}", ha='center', va='center',
             fontsize=22, fontweight='bold', color=TEXT_MAIN)
    ax0.text(0, -0.22, "samples", ha='center', va='center',
             fontsize=9, color=TEXT_DIM, fontfamily='DM Mono')

    # Custom legend
    handles = [mpatches.Patch(facecolor=c, label=l, linewidth=0)
               for c, l in zip(colors, labels)]
    ax0.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.08),
               ncol=2, frameon=False, labelcolor=TEXT_MAIN)
    ax0.set_title("Potability Split", color=TEXT_MAIN)

    # ── Annotated horizontal bar ───────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1])
    ax1.set_facecolor(CARD_BG)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_color(GRID_COLOR)

    y_pos = [1, 0]
    bar_colors = [ACCENT_RED, ACCENT_TEAL]
    for i, (yi, cnt, col) in enumerate(zip(y_pos, counts.values, bar_colors)):
        ax1.barh(yi, cnt, height=0.5, color=col, alpha=0.85,
                 edgecolor='none')
        ax1.text(cnt + 18, yi, f"{cnt:,}", va='center',
                 fontsize=12, fontweight='bold', color=col)
        ax1.text(-30, yi, labels[i], va='center', ha='right',
                 fontsize=10, color=TEXT_DIM)

    pct_potable = counts[1] / len(df) * 100
    ax1.set_xlim(-250, counts.max() * 1.18)
    ax1.set_yticks([])
    ax1.set_xlabel("Count", color=TEXT_DIM, fontsize=9)
    ax1.set_title("Class Counts", color=TEXT_MAIN)
    ax1.grid(axis='y', visible=False)

    ax1.text(counts.max() * 1.1, -0.6,
             f"Imbalance ratio  {counts[0]/counts[1]:.2f}:1",
             ha='right', fontsize=8, color=TEXT_DIM, fontstyle='italic')

    fig.suptitle("Target Variable — Water Potability",
                 fontsize=15, fontweight='bold', color=TEXT_MAIN, y=1.02)
    return fig


def fig_feature_distributions(df, feature):
    """Styled histogram with KDE overlay for one feature."""
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=DARK_BG)

    data_np = df[feature].dropna().values
    ax.hist(data_np, bins=40, color=ACCENT_TEAL, alpha=0.18,
            edgecolor='none', density=True)

    # KDE per class
    for val, col, lbl in [(0, ACCENT_RED, 'Not Potable'), (1, ACCENT_TEAL, 'Potable')]:
        subset = df[df['Potability'] == val][feature].dropna()
        kde_x  = np.linspace(data_np.min(), data_np.max(), 300)
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(subset)
        ax.plot(kde_x, kde(kde_x), color=col, lw=2, label=lbl)
        ax.fill_between(kde_x, kde(kde_x), alpha=0.10, color=col)

    mean_v   = df[feature].mean()
    median_v = df[feature].median()
    ax.axvline(mean_v,   color=ACCENT_GOLD, lw=1.5, linestyle='--',
               label=f'Mean  {mean_v:.2f}')
    ax.axvline(median_v, color='#a29bfe',   lw=1.5, linestyle=':',
               label=f'Median {median_v:.2f}')

    unit = FEATURE_UNITS.get(feature, '')
    ax.set_xlabel(f"{feature}  [{unit}]", color=TEXT_DIM)
    ax.set_ylabel("Density", color=TEXT_DIM)
    ax.set_title(f"Distribution — {feature}", color=TEXT_MAIN)
    ax.legend()
    fig.tight_layout()
    return fig


def fig_confusion_matrix(cm):
    """Annotated confusion matrix with accuracy-styled cells."""
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=DARK_BG)

    # Custom color: TP/TN green, FP/FN red
    cell_colors = np.array([
        [ACCENT_RED,  "#2d3748"],
        ["#2d3748",  ACCENT_TEAL]
    ])
    # Draw cells manually
    class_labels = ['Not\nPotable', 'Potable']
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            color = ACCENT_TEAL if i == j else ACCENT_RED
            alpha = 0.22 if i == j else 0.12
            rect = FancyBboxPatch((j - 0.42, i - 0.42), 0.84, 0.84,
                                  boxstyle="round,pad=0.04",
                                  facecolor=color, alpha=alpha,
                                  edgecolor=color, linewidth=1.5)
            ax.add_patch(rect)
            pct = cm[i, j] / total * 100
            ax.text(j, i, f"{cm[i, j]:,}\n({pct:.1f}%)",
                    ha='center', va='center', fontsize=13, fontweight='bold',
                    color=ACCENT_TEAL if i == j else ACCENT_RED)

    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(-0.55, 1.55)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(class_labels, fontsize=10, color=TEXT_DIM)
    ax.set_yticklabels(class_labels, fontsize=10, color=TEXT_DIM)
    ax.set_xlabel("Predicted", labelpad=10)
    ax.set_ylabel("Actual",    labelpad=10)
    ax.set_title("Confusion Matrix", pad=16)
    ax.grid(False)

    # Axis labels
    ax.text(0.5, -0.54, "Predicted Class", ha='center', fontsize=9,
            color=TEXT_DIM, transform=ax.transData)
    fig.tight_layout()
    return fig


def fig_roc_curve(y_test, y_prob):
    """Filled ROC curve with operating point."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5), facecolor=DARK_BG)
    ax.plot([0, 1], [0, 1], '--', color=TEXT_DIM, lw=1, label='Random')
    ax.plot(fpr, tpr, color=ACCENT_TEAL, lw=2.5,
            label=f'AUC = {auc:.4f}')
    ax.fill_between(fpr, tpr, alpha=0.12, color=ACCENT_TEAL)

    # Best threshold dot (closest to top-left)
    dist = np.sqrt(fpr**2 + (1 - tpr)**2)
    best_idx = np.argmin(dist)
    ax.scatter(fpr[best_idx], tpr[best_idx], color=ACCENT_GOLD,
               s=80, zorder=5, label=f'Best threshold\n(FPR={fpr[best_idx]:.2f}, TPR={tpr[best_idx]:.2f})')

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", pad=14)
    ax.legend(loc='lower right')
    fig.tight_layout()
    return fig


def fig_feature_importance(importance_df):
    """Horizontal lollipop chart for feature importance."""
    df_plot = importance_df.sort_values('importance')
    n = len(df_plot)

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=DARK_BG)
    y_pos = np.arange(n)

    # FIX 2: Updated for Matplotlib 3.9+ compatibility
    cmap  = plt.get_cmap('cool')
    cols  = [cmap(i / (n - 1)) for i in range(n)]

    # Stems
    for yi, val, col in zip(y_pos, df_plot['importance'], cols):
        ax.plot([0, val], [yi, yi], color=col, lw=1.8, alpha=0.5)
        ax.scatter(val, yi, color=col, s=120, zorder=4, edgecolors='white', linewidths=0.6)
        ax.text(val + 0.002, yi, f"{val:.4f}", va='center',
                fontsize=8.5, color=col, fontfamily='DM Mono')

    # Cumulative line (secondary axis)
    cum = df_plot['importance'].cumsum() / df_plot['importance'].sum() * 100
    ax2 = ax.twiny()
    ax2.plot(cum.values, y_pos, color=ACCENT_GOLD, lw=1.4,
             linestyle='--', alpha=0.7, label='Cumulative %')
    ax2.axvline(80, color=ACCENT_GOLD, lw=0.8, linestyle=':', alpha=0.5)
    ax2.set_xlabel("Cumulative Importance %", color=ACCENT_GOLD, fontsize=9)
    ax2.tick_params(colors=ACCENT_GOLD)
    ax2.set_xlim(0, 110)
    ax2.spines['top'].set_color(ACCENT_GOLD)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot['feature'], fontsize=10)
    ax.set_xlabel("Gini Importance", fontsize=10)
    ax.set_title("Feature Importance  ·  Random Forest", pad=14)
    ax.set_xlim(-0.01, df_plot['importance'].max() * 1.22)
    ax.grid(axis='x', alpha=0.4)
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    return fig


def fig_correlation_heatmap(df):
    """Triangular heatmap with diverging coolwarm palette."""
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=DARK_BG)

    sns.heatmap(
        corr, mask=mask, ax=ax,
        annot=True, fmt='.2f', annot_kws={'size': 9, 'color': TEXT_MAIN},
        cmap='coolwarm', vmin=-1, vmax=1, center=0,
        linewidths=1.2, linecolor=DARK_BG,
        square=True, cbar_kws={'shrink': 0.75}
    )
    ax.set_title("Feature Correlation Matrix", pad=16)
    ax.tick_params(axis='x', rotation=40, labelsize=9, colors=TEXT_DIM)
    ax.tick_params(axis='y', rotation=0,  labelsize=9, colors=TEXT_DIM)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors=TEXT_DIM, labelsize=8)
    cbar.ax.yaxis.label.set_color(TEXT_DIM)
    fig.tight_layout()
    return fig


def fig_all_distributions(df):
    """3×3 KDE grid for all 9 features."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), facecolor=DARK_BG)
    axes = axes.ravel()

    from scipy.stats import gaussian_kde
    for i, feat in enumerate(FEATURE_NAMES):
        ax = axes[i]
        for val, col, lbl in [(0, ACCENT_RED, 'Not Potable'), (1, ACCENT_TEAL, 'Potable')]:
            data = df[df['Potability'] == val][feat].dropna()
            kde_x = np.linspace(df[feat].min(), df[feat].max(), 300)
            kde   = gaussian_kde(data)
            ax.plot(kde_x, kde(kde_x), color=col, lw=2)
            ax.fill_between(kde_x, kde(kde_x), alpha=0.12, color=col)
        ax.set_title(feat, fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel(FEATURE_UNITS.get(feat, ''), fontsize=8, color=TEXT_DIM)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)

    # Shared legend
    handles = [
        mpatches.Patch(color=ACCENT_RED,  label='Not Potable'),
        mpatches.Patch(color=ACCENT_TEAL, label='Potable'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02),
               frameon=False, labelcolor=TEXT_MAIN, fontsize=10)
    fig.suptitle("Feature Distributions by Class", fontsize=15,
                 fontweight='bold', color=TEXT_MAIN, y=1.01)
    fig.tight_layout()
    return fig


def fig_classification_report(y_test, y_pred):
    """Visual classification report as a styled table chart."""
    from sklearn.metrics import classification_report
    report = classification_report(y_test, y_pred,
                                   target_names=['Not Potable', 'Potable'],
                                   output_dict=True)
    rows   = ['Not Potable', 'Potable', 'macro avg', 'weighted avg']
    cols   = ['precision', 'recall', 'f1-score']
    data   = np.array([[report[r][c] for c in cols] for r in rows])

    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(0, 1); ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.axis('off')

    col_colors = [ACCENT_TEAL, ACCENT_GOLD, '#a29bfe']
    col_x      = [0.28, 0.55, 0.80]
    row_y      = [3, 2, 1, 0]

    # Headers
    for cx, lbl, col in zip(col_x, ['Precision', 'Recall', 'F1-Score'], col_colors):
        ax.text(cx, 3.65, lbl, ha='center', va='center',
                fontsize=10, fontweight='bold', color=col, fontfamily='DM Mono')

    ax.axhline(3.35, color=GRID_COLOR, lw=1)

    for ri, (row_lbl, yi) in enumerate(zip(rows, row_y)):
        bg = CARD_BG if ri % 2 == 0 else "#1a2030"
        rect = FancyBboxPatch((-0.01, yi - 0.42), 1.02, 0.84,
                               boxstyle="square,pad=0", facecolor=bg,
                               edgecolor='none', transform=ax.transData)
        ax.add_patch(rect)
        ax.text(0.08, yi, row_lbl, ha='left', va='center',
                fontsize=10, color=TEXT_MAIN)
        for cx, val, col in zip(col_x, data[ri], col_colors):
            bar_w = val * 0.16
            ax.barh(yi, bar_w, left=cx - 0.08, height=0.3,
                    color=col, alpha=0.20, align='center')
            ax.text(cx, yi, f"{val:.3f}", ha='center', va='center',
                    fontsize=11, fontweight='bold', color=col, fontfamily='DM Mono')

    ax.set_title("Classification Report", pad=14,
                 fontsize=13, fontweight='bold', color=TEXT_MAIN)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Pages
# ══════════════════════════════════════════════════════════════════════════

def page_data_overview(df, predictor):
    st.markdown("## 📊 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    pot = int(df['Potability'].sum())
    c1.metric("Total Samples",   f"{len(df):,}")
    c2.metric("Features",        len(FEATURE_NAMES))
    c3.metric("Potable",         f"{pot:,}  ({pot/len(df)*100:.1f}%)")
    c4.metric("Not Potable",     f"{len(df)-pot:,}  ({(1-pot/len(df))*100:.1f}%)")

    st.markdown("---")
    st.markdown("### Target Distribution")
    st.pyplot(fig_target_distribution(df), use_container_width=True)

    st.markdown("---")
    st.markdown("### Feature Explorer")
    feat = st.selectbox("Select a feature:", FEATURE_NAMES,
                        format_func=lambda f: f"{f}  [{FEATURE_UNITS[f]}]")
    st.pyplot(fig_feature_distributions(df, feat), use_container_width=True)

    st.markdown("---")
    st.markdown("### All Feature Distributions")
    st.pyplot(fig_all_distributions(df), use_container_width=True)

    st.markdown("---")
    st.markdown("### Statistical Summary")
    st.dataframe(df.describe().T.style
                 .background_gradient(cmap='Blues', subset=['mean', 'std'])
                 .format('{:.3f}'), use_container_width=True)


def page_model_training(predictor, df):
    st.markdown("## 🤖 Model Training")
    st.markdown(f"Training a `RandomForestClassifier` with **200 estimators** on an 80/20 stratified split.")

    if st.button("🚀 Train Model"):
        with st.spinner("Training…"):
            X_train, X_test, y_train, y_test, y_pred, y_prob = predictor.train_model(df)
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            cm  = confusion_matrix(y_test, y_pred)

        st.success("✅ Model trained successfully!")
        joblib.dump(predictor.model, 'water_quality_model.pkl')

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test Accuracy",  f"{acc:.4f}")
        m2.metric("ROC-AUC",        f"{auc:.4f}")
        m3.metric("Train samples",  f"{len(X_train):,}")
        m4.metric("Test samples",   f"{len(X_test):,}")

        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Confusion Matrix")
            st.pyplot(fig_confusion_matrix(cm), use_container_width=True)
        with col_b:
            st.markdown("#### ROC Curve")
            st.pyplot(fig_roc_curve(y_test, y_prob), use_container_width=True)

        st.markdown("---")
        st.markdown("#### Classification Report")
        st.pyplot(fig_classification_report(y_test, y_pred), use_container_width=True)

        st.info("💾 Model saved as `water_quality_model.pkl`")


def page_prediction(predictor, df):
    st.markdown("## 🔮 Predict Water Quality")

    try:
        predictor.model = joblib.load('water_quality_model.pkl')
        st.success("✅ Pre-trained model loaded.")
    except FileNotFoundError:
        st.warning("⚠️ No saved model. Train one first.")
        if st.button("Train Now"):
            with st.spinner("Training…"):
                predictor.train_model(df)
                joblib.dump(predictor.model, 'water_quality_model.pkl')
            st.success("✅ Done!")
            # FIX 3: Force a script rerun to render the newly unlocked sliders
            st.rerun() 
        return

    st.markdown("### Enter Water Parameters")

    slider_cfg = {
        'ph':               (0.0,  14.0,   7.0, 0.1),
        'Hardness':         (0.0, 500.0, 196.0, 1.0),
        'Solids':           (0.0, 60000.0, 22000.0, 100.0),
        'Chloramines':      (0.0,  15.0,   7.0, 0.1),
        'Sulfate':          (0.0, 500.0, 333.0, 1.0),
        'Conductivity':     (0.0, 800.0, 426.0, 1.0),
        'Organic_carbon':   (0.0,  30.0,  14.0, 0.1),
        'Trihalomethanes':  (0.0, 130.0,  66.0, 0.1),
        'Turbidity':        (0.0,  10.0,   4.0, 0.1),
    }

    c1, c2, c3 = st.columns(3)
    feats  = FEATURE_NAMES
    inputs = {}
    cols   = [c1, c2, c3]
    for i, feat in enumerate(feats):
        mn, mx, dv, step = slider_cfg[feat]
        inputs[feat] = cols[i % 3].slider(
            f"{feat}  [{FEATURE_UNITS[feat]}]", mn, mx, dv, step)

    if st.button("🔍 Predict", type="primary"):
        vals = [inputs[f] for f in FEATURE_NAMES]
        pred, prob = predictor.predict_water_quality(vals)

        st.markdown("---")
        if pred == 1:
            st.markdown(f"""
            <div class="card-potable">
                <div class="card-title" style="color:{ACCENT_TEAL}">✅  Potable Water</div>
                <div class="card-sub">This water sample is predicted to be <strong>safe for drinking</strong>.</div>
                <br/>
                <span style="font-family:'DM Mono',monospace;font-size:1.1rem;color:{ACCENT_TEAL};font-weight:700">
                    Confidence: {prob[1]:.2%}
                </span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="card-not-potable">
                <div class="card-title" style="color:{ACCENT_RED}">❌  Not Potable</div>
                <div class="card-sub">This water sample is predicted to be <strong>unsafe for drinking</strong>.</div>
                <br/>
                <span style="font-family:'DM Mono',monospace;font-size:1.1rem;color:{ACCENT_RED};font-weight:700">
                    Confidence: {prob[0]:.2%}
                </span>
            </div>""", unsafe_allow_html=True)

        # Probability gauge
        st.markdown("#### Probability Breakdown")
        fig, ax = plt.subplots(figsize=(8, 2.2), facecolor=DARK_BG)
        categories = ['Not Potable', 'Potable']
        bar_colors = [ACCENT_RED, ACCENT_TEAL]
        for yi, (lbl, p, col) in enumerate(zip(categories, prob, bar_colors)):
            ax.barh(yi, p, height=0.45, color=col, alpha=0.85, edgecolor='none')
            ax.barh(yi, 1, height=0.45, color=col, alpha=0.07, edgecolor='none')
            ax.text(p + 0.02, yi, f"{p:.2%}", va='center',
                    fontsize=13, fontweight='bold', color=col, fontfamily='DM Mono')
            ax.text(-0.02, yi, lbl, va='center', ha='right',
                    fontsize=10, color=TEXT_DIM)
        ax.set_xlim(-0.25, 1.25)
        ax.set_yticks([])
        ax.axvline(0.5, color=TEXT_DIM, lw=0.8, linestyle=':')
        ax.set_xlabel("Probability", color=TEXT_DIM, fontsize=9)
        ax.grid(axis='y', visible=False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)


def page_model_analysis(predictor, df):
    st.markdown("## 📈 Model Analysis")

    try:
        predictor.model = joblib.load('water_quality_model.pkl')
    except FileNotFoundError:
        st.warning("⚠️ Train the model first.")
        return

    st.markdown("### Feature Importance")
    imp_df = predictor.get_feature_importance()
    st.pyplot(fig_feature_importance(imp_df), use_container_width=True)

    st.markdown("---")
    st.markdown("### Correlation Heatmap")
    st.pyplot(fig_correlation_heatmap(df), use_container_width=True)

    st.markdown("---")
    st.markdown("### Importance Table")
    imp_df['Importance %']  = (imp_df['importance'] * 100).round(2)
    imp_df['Cumulative %']  = imp_df['Importance %'].cumsum().round(2)
    st.dataframe(
        imp_df.rename(columns={'feature': 'Feature', 'importance': 'Gini Importance'})
        .style.background_gradient(cmap='Greens', subset=['Gini Importance'])
        .format({'Gini Importance': '{:.4f}', 'Importance %': '{:.2f}%', 'Cumulative %': '{:.2f}%'}),
        use_container_width=True
    )


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    predictor = WaterQualityPredictor()

    # Sidebar
    with st.sidebar:
        st.markdown(f"""
        <div style='padding:4px 0 20px'>
            <div style='font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;
                        color:{ACCENT_TEAL};letter-spacing:-0.5px'>💧 WaterIQ</div>
            <div style='font-family:DM Mono,monospace;font-size:0.68rem;
                        color:{TEXT_DIM};letter-spacing:2px'>POTABILITY PREDICTOR</div>
        </div>
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigate",
            ["📊 Data Overview", "🤖 Model Training",
             "🔮 Make Prediction", "📈 Model Analysis"],
            label_visibility='collapsed'
        )
        st.markdown("---")
        st.markdown(f"<div style='font-size:0.72rem;color:{TEXT_DIM};line-height:1.7'>"
                    "Dataset: <code>potability.csv</code><br>"
                    "Model: Random Forest (n=200)<br>"
                    "Split: 80 / 20  ·  stratified</div>", unsafe_allow_html=True)

    # Header
    st.markdown('<div class="app-title">💧 Water Quality Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">AI-powered potability assessment</div>', unsafe_allow_html=True)

    df = predictor.load_data()
    if df is None:
        return

    if page == "📊 Data Overview":
        page_data_overview(df, predictor)
    elif page == "🤖 Model Training":
        page_model_training(predictor, df)
    elif page == "🔮 Make Prediction":
        page_prediction(predictor, df)
    elif page == "📈 Model Analysis":
        page_model_analysis(predictor, df)


if __name__ == "__main__":
    main()