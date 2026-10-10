
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Customer Insights | Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# PROFESSIONAL CSS
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --navy: #10243A;
    --navy-light: #193650;
    --teal: #0D9488;
    --teal-light: #CCFBF1;
    --text: #172B4D;
    --muted: #64748B;
    --border: #E2E8F0;
    --surface: #FFFFFF;
    --background: #F4F7FB;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: var(--background);
    color: var(--text);
}

header[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu, footer {
    visibility: hidden;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

h1, h2, h3, p, label {
    color: var(--text);
}

h1 {
    font-weight: 800 !important;
    letter-spacing: -1px;
}

h2, h3 {
    font-weight: 700 !important;
}

[data-testid="stSidebar"] {
    background: var(--navy);
    border-right: 1px solid #29435C;
}

[data-testid="stSidebar"] * {
    color: #F1F5F9 !important;
}

[data-testid="stSidebar"] .stCaption {
    color: #B8C8D8 !important;
}

[data-testid="stSidebar"] hr {
    border-color: #36506A;
}

[data-testid="stSidebar"] input {
    color: #10243A !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #10243A !important;
}

[data-testid="stSidebar"] [data-testid="stSlider"] {
    padding-top: 0.5rem;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid var(--border);
    padding: 18px 20px;
    border-radius: 14px;
    box-shadow: 0 3px 12px rgba(16,36,58,0.035);
}

[data-testid="stMetricLabel"] {
    color: #64748B !important;
    font-size: 0.83rem !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    color: #10243A !important;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
}

.panel {
    background: white;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 14px;
}

.hero {
    background: linear-gradient(120deg, #10243A, #193F57);
    border-radius: 18px;
    padding: 25px 28px;
    margin-bottom: 22px;
    color: white;
}

.hero h1, .hero p, .hero span {
    color: white !important;
}

.hero p {
    margin-bottom: 0;
    color: #D4E3EE !important;
    font-size: 0.95rem;
}

.eyebrow {
    text-transform: uppercase;
    letter-spacing: 1.7px;
    font-size: 0.72rem;
    font-weight: 700;
    color: #5EEAD4 !important;
    margin-bottom: 8px;
}

.section-title {
    font-size: 1.12rem;
    font-weight: 750;
    margin-bottom: 3px;
    color: #10243A;
}

.section-subtitle {
    color: #64748B;
    font-size: 0.84rem;
    margin-bottom: 15px;
}

.pill {
    display: inline-block;
    padding: 6px 11px;
    border-radius: 30px;
    background: #CCFBF1;
    color: #115E59;
    font-size: 0.76rem;
    font-weight: 700;
}

.stButton > button,
.stDownloadButton > button {
    border-radius: 9px;
    border: 1px solid #0D9488;
    background: #0D9488;
    color: white;
    font-weight: 650;
    min-height: 42px;
    transition: 0.2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #0F766E;
    border-color: #0F766E;
    color: white;
}

.stTextInput input,
.stNumberInput input {
    border-radius: 9px;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
}

div[data-testid="stForm"] {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    padding: 16px;
    border-radius: 12px;
}

hr {
    border-color: #E2E8F0;
}

@media (max-width: 768px) {
    .block-container {
        padding: 1rem;
    }
    .hero {
        padding: 20px;
    }
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# HELPER FUNCTIONS
# =====================================================
@st.cache_data
def load_dataset(uploaded_bytes=None):
    if uploaded_bytes is not None:
        from io import BytesIO
        return pd.read_csv(BytesIO(uploaded_bytes))
    return pd.read_csv("dataset.csv")


def make_segment_name(income, spending, income_mid, spending_mid):
    if income >= income_mid and spending >= spending_mid:
        return "High-value"
    elif income >= income_mid and spending < spending_mid:
        return "Affluent, low-spending"
    elif income < income_mid and spending >= spending_mid:
        return "Lower-income, high-spending"
    return "Low-income, low-spending"


# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("""
    <div style="padding:10px 0 20px 0;">
        <div style="font-size:2rem;">◈</div>
        <div style="font-size:1.2rem;font-weight:800;">
            INSIGHT<span style="color:#5EEAD4;">IQ</span>
        </div>
        <div style="font-size:0.76rem;color:#B8C8D8;">
            CUSTOMER ANALYTICS PLATFORM
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### Workspace")
    st.caption("Customer segmentation")

    uploaded_file = st.file_uploader(
        "Upload customer CSV (optional)",
        type=["csv"],
        help="Leave empty to use dataset.csv in your project folder."
    )

    st.divider()
    st.markdown("### Model settings")

    k = st.slider(
        "Number of clusters (K)",
        min_value=2,
        max_value=10,
        value=5,
        help="Select how many customer groups K-Means should create."
    )

    st.caption("Algorithm: K-Means Clustering")
    st.caption("Scaling: StandardScaler")
    st.caption("Initialization: K-Means++")

    st.divider()

    st.markdown("### About")
    st.caption(
        "Discover customer patterns, compare segments, "
        "and explore data-driven marketing opportunities."
    )

    st.markdown("""
    <div style="background:#193650;padding:12px;border-radius:10px;
    font-size:0.78rem;color:#D4E3EE;">
        <b style="color:#5EEAD4;">●</b> Analytics engine ready
        <br><br>Unsupervised Machine Learning
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# LOAD AND VALIDATE DATA
# =====================================================
try:
    if uploaded_file is not None:
        raw_df = load_dataset(uploaded_file.getvalue())
    else:
        raw_df = load_dataset()
except FileNotFoundError:
    st.error(
        "dataset.csv was not found. Place it in the same folder "
        "as app.py, or upload a CSV using the sidebar."
    )
    st.stop()
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

feature_columns = [
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

missing = [c for c in feature_columns if c not in raw_df.columns]

if missing:
    st.error(
        "Your dataset is missing required columns: "
        + ", ".join(missing)
    )
    st.info(
        "Required columns: Annual Income (k$), "
        "Spending Score (1-100)."
    )
    st.stop()

df = raw_df.copy()
df = df.drop_duplicates()

for col in feature_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=feature_columns).copy()

if len(df) < 3:
    st.error("At least 3 valid customer records are required.")
    st.stop()

if k >= len(df):
    st.error(
        f"K must be smaller than the number of valid records ({len(df)}). "
        "Please upload more data or reduce K."
    )
    st.stop()

# Create a display identifier if the input lacks CustomerID.
if "CustomerID" not in df.columns:
    df["CustomerID"] = range(1, len(df) + 1)

if "Age" not in df.columns:
    df["Age"] = np.nan
else:
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")


# =====================================================
# CLUSTERING PIPELINE
# =====================================================
features = df[feature_columns]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(scaled_features)

# Use cluster numbers only as identifiers.
centers = scaler.inverse_transform(kmeans.cluster_centers_)

# Silhouette score is undefined if all observations belong
# to one cluster or every observation becomes its own cluster.
if 1 < df["Cluster"].nunique() < len(df):
    sil_score = silhouette_score(scaled_features, df["Cluster"])
else:
    sil_score = np.nan

summary = df.groupby("Cluster").agg(
    Customers=("CustomerID", "count"),
    Avg_Income=("Annual Income (k$)", "mean"),
    Avg_Spending=("Spending Score (1-100)", "mean"),
    Avg_Age=("Age", "mean")
).round(2)

summary["Customer Share (%)"] = (
    summary["Customers"] / len(df) * 100
).round(1)

income_mid = df["Annual Income (k$)"].median()
spending_mid = df["Spending Score (1-100)"].median()

summary["Segment"] = [
    make_segment_name(
        row["Avg_Income"],
        row["Avg_Spending"],
        income_mid,
        spending_mid
    )
    for _, row in summary.iterrows()
]


# =====================================================
# HERO HEADER
# =====================================================
st.markdown(f"""
<div class="hero">
    <div class="eyebrow">CUSTOMER INTELLIGENCE / OVERVIEW</div>
    <h1 style="margin:0 0 8px 0;font-size:2rem;">
        Customer Segmentation
    </h1>
    <p>
        Understand your customers. Discover meaningful groups.
        Make smarter marketing decisions.
    </p>
    <div style="margin-top:17px;">
        <span class="pill">K-Means Clustering</span>
        <span style="margin-left:8px;font-size:0.8rem;">
            {len(df):,} valid customer records
        </span>
    </div>
</div>
""", unsafe_allow_html=True)


# =====================================================
# KPI CARDS
# =====================================================
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "TOTAL CUSTOMERS",
        f"{len(df):,}",
        help="Number of valid customer records used for clustering."
    )

with c2:
    st.metric(
        "CUSTOMER SEGMENTS",
        f"{df['Cluster'].nunique():,}",
        help="Number of clusters selected in the sidebar."
    )

with c3:
    st.metric(
        "SILHOUETTE SCORE",
        f"{sil_score:.3f}" if pd.notna(sil_score) else "N/A",
        help="Higher is generally better; interpret in context."
    )

with c4:
    st.metric(
        "AVG. SPENDING SCORE",
        f"{df['Spending Score (1-100)'].mean():.1f}",
        help="Mean spending score across the dataset."
    )

st.write("")


# =====================================================
# CHARTS ROW
# =====================================================
chart_col, mix_col = st.columns([1.65, 1])

with chart_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Customer cluster map</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-subtitle">'
        'Income vs. spending behavior · Each color represents a cluster'
        '</div>',
        unsafe_allow_html=True
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFCFE")

    cmap = plt.get_cmap("viridis", k)

    for cluster_id in sorted(df["Cluster"].unique()):
        subset = df[df["Cluster"] == cluster_id]
        ax.scatter(
            subset["Annual Income (k$)"],
            subset["Spending Score (1-100)"],
            s=65,
            alpha=0.82,
            color=cmap(cluster_id),
            edgecolors="white",
            linewidths=0.7,
            label=f"Cluster {cluster_id}"
        )

    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="X",
        s=210,
        color="#F97316",
        edgecolors="white",
        linewidths=1.2,
        label="Centroids",
        zorder=5
    )

    ax.set_xlabel("Annual income (thousands)", fontsize=9)
    ax.set_ylabel("Spending score", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#DCE5EF")
    ax.tick_params(colors="#64748B", labelsize=8)
    ax.legend(
        frameon=False,
        fontsize=8,
        ncol=2,
        loc="best"
    )
    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)

with mix_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Segment distribution</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-subtitle">'
        'Share of customers in each group'
        '</div>',
        unsafe_allow_html=True
    )

    counts = df["Cluster"].value_counts().sort_index()

    fig2, ax2 = plt.subplots(figsize=(5, 4.5))
    fig2.patch.set_facecolor("white")

    bars = ax2.bar(
        [f"C{i}" for i in counts.index],
        counts.values,
        color=[cmap(i) for i in counts.index],
        width=0.62
    )

    ax2.set_ylabel("Customers", fontsize=9)
    ax2.grid(axis="y", linestyle="--", alpha=0.2)
    ax2.set_axisbelow(True)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.spines[["left", "bottom"]].set_color("#DCE5EF")
    ax2.tick_params(colors="#64748B", labelsize=8)

    for bar, count in zip(bars, counts.values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#10243A"
        )

    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    st.markdown('</div>', unsafe_allow_html=True)


# =====================================================
# SEGMENT SUMMARY
# =====================================================
st.markdown(
    '<div class="section-title">Segment performance</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="section-subtitle">'
    'Compare the size and average behavior of every segment'
    '</div>',
    unsafe_allow_html=True
)

display_summary = summary.rename(columns={
    "Avg_Income": "Average Income (k$)",
    "Avg_Spending": "Average Spending Score",
    "Avg_Age": "Average Age"
})

st.dataframe(
    display_summary,
    use_container_width=True,
    hide_index=False
)


# =====================================================
# CUSTOMER EXPLORER
# =====================================================
st.write("")
with st.expander("Explore customer records", expanded=False):
    f1, f2 = st.columns([1, 1])

    with f1:
        chosen_cluster = st.selectbox(
            "Filter by cluster",
            options=["All"] + sorted(df["Cluster"].unique().tolist())
        )

    with f2:
        search_id = st.text_input(
            "Search customer ID (optional)",
            placeholder="Enter customer ID"
        )

    filtered = df.copy()

    if chosen_cluster != "All":
        filtered = filtered[filtered["Cluster"] == chosen_cluster]

    if search_id.strip():
        filtered = filtered[
            filtered["CustomerID"].astype(str).str.contains(
                search_id.strip(),
                case=False,
                na=False
            )
        ]

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True
    )

    st.caption(f"Showing {len(filtered)} customer records.")


# =====================================================
# NEW CUSTOMER ASSIGNMENT
# =====================================================
st.markdown("---")
st.markdown(
    '<div class="section-title">Find a customer segment</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="section-subtitle">'
    'Enter a new customer’s income and spending score to assign '
    'them to the nearest learned cluster.'
    '</div>',
    unsafe_allow_html=True
)

predict_col, result_col = st.columns([1, 1])

with predict_col:
    with st.form("customer_prediction"):
        income = st.number_input(
            "Annual income (thousands)",
            min_value=0.0,
            max_value=1000.0,
            value=50.0,
            step=1.0
        )

        spending = st.number_input(
            "Spending score",
            min_value=1,
            max_value=100,
            value=50,
            step=1
        )

        predict_button = st.form_submit_button(
            "Analyze customer",
            use_container_width=True
        )

with result_col:
    if predict_button:
        new_scaled = scaler.transform([[income, spending]])
        predicted_cluster = int(kmeans.predict(new_scaled)[0])
        details = summary.loc[predicted_cluster]

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="eyebrow" style="color:#0D9488 !important;">'
            f'CUSTOMER ANALYSIS RESULT</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div style="font-size:1.5rem;font-weight:800;'
            f'color:#10243A;">Cluster {predicted_cluster}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<span class="pill">{details["Segment"]}</span>',
            unsafe_allow_html=True
        )

        st.write("")
        r1, r2 = st.columns(2)
        r1.metric("Customers in segment", int(details["Customers"]))
        r2.metric("Customer share", f'{details["Customer Share (%)"]:.1f}%')

        st.caption(
            "Segment interpretation is based on income and spending "
            "relative to dataset medians. This is a cluster assignment, "
            "not a purchase prediction."
        )
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="panel" style="min-height:185px;">
            <div class="section-title">Ready to analyze</div>
            <p style="color:#64748B;font-size:0.9rem;margin-top:12px;">
                Submit the customer details to see the assigned cluster,
                segment interpretation, and group statistics.
            </p>
        </div>
        """, unsafe_allow_html=True)


# =====================================================
# DOWNLOAD RESULTS
# =====================================================
st.markdown("---")

download_col, note_col = st.columns([1, 2])

with download_col:
    st.download_button(
        label="⬇ Download segmented customers",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="segmented_customers.csv",
        mime="text/csv",
        use_container_width=True
    )

with note_col:
    st.caption(
        "Method: K-Means clustering on standardized annual income "
        "and spending score. Cluster numbers are identifiers, not rankings. "
        "Review cluster statistics before making marketing decisions."
    )

st.markdown("""
<div style="text-align:center;padding:18px 0 4px 0;
color:#94A3B8;font-size:0.76rem;">
    INSIGHTIQ · CUSTOMER ANALYTICS · POWERED BY PYTHON & SCIKIT-LEARN
</div>
""", unsafe_allow_html=True)
