import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

RAW_DATA_PATH = "data/retail_transactions.csv"
OUTPUT_DIR = "outputs"

plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# DATA CLEANING
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["InvoiceDate"])

    before = len(df)
    df = df.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]

    df["CustomerID"] = df["CustomerID"].astype(int)
    df["LineTotal"] = df["Quantity"] * df["UnitPrice"]

    after = len(df)
    print(f"[clean] Removed {before - after:,} rows (cancellations / missing IDs / invalid values). "
          f"{after:,} clean line items remain.")
    return df

# EDA
def run_eda(df: pd.DataFrame):
    print("\n[EDA] Dataset overview")
    print(f"  Date range        : {df['InvoiceDate'].min().date()} -> {df['InvoiceDate'].max().date()}")
    print(f"  Unique customers   : {df['CustomerID'].nunique():,}")
    print(f"  Unique invoices    : {df['InvoiceNo'].nunique():,}")
    print(f"  Unique products    : {df['StockCode'].nunique():,}")
    print(f"  Total revenue      : {df['LineTotal'].sum():,.2f}")
    print(f"  Avg order value    : {df.groupby('InvoiceNo')['LineTotal'].sum().mean():,.2f}")

    # Monthly revenue trend
    monthly = df.set_index("InvoiceDate").resample("ME")["LineTotal"].sum()
    fig, ax = plt.subplots(figsize=(8, 4))
    monthly.plot(ax=ax, marker="o", color="#2E5EAA")
    ax.set_title("Monthly Revenue Trend")
    ax.set_ylabel("Revenue")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/eda_monthly_revenue.png")
    plt.close(fig)

    # Revenue by country
    by_country = df.groupby("Country")["LineTotal"].sum().sort_values(ascending=False).head(8)
    fig, ax = plt.subplots(figsize=(8, 4))
    by_country.plot(kind="bar", ax=ax, color="#3E8E7E")
    ax.set_title("Revenue by Country (Top 8)")
    ax.set_ylabel("Revenue")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/eda_revenue_by_country.png")
    plt.close(fig)

    # Top 10 products by revenue
    by_product = df.groupby("Description")["LineTotal"].sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8, 4))
    by_product.sort_values().plot(kind="barh", ax=ax, color="#C1666B")
    ax.set_title("Top 10 Products by Revenue")
    ax.set_xlabel("Revenue")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/eda_top_products.png")
    plt.close(fig)

    print(f"[EDA] Saved 3 charts to {OUTPUT_DIR}/")

# RFM 
def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    snapshot_date = df["InvoiceDate"].max() + timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("LineTotal", "sum"),
    ).reset_index()

    print("\n[RFM] Summary statistics")
    print(rfm[["Recency", "Frequency", "Monetary"]].describe().round(2))
    return rfm


# K-Means
def find_best_k(X_scaled: np.ndarray, k_range=range(3, 7)) -> int:
    inertias, sil_scores = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(list(k_range), inertias, marker="o", color="#2E5EAA")
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(list(k_range), sil_scores, marker="o", color="#C1666B")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Score")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/kmeans_k_selection.png")
    plt.close(fig)

    best_k = list(k_range)[int(np.argmax(sil_scores))]
    print(f"\n[KMeans] Silhouette scores by k: "
          f"{dict(zip(k_range, np.round(sil_scores, 3)))}")
    print(f"[KMeans] Selected k = {best_k} (highest silhouette score)")
    return best_k


def run_clustering(rfm: pd.DataFrame) -> pd.DataFrame:
    features = rfm[["Recency", "Frequency", "Monetary"]].copy()

    # Log-transform Monetary & Frequency to reduce skew 
    features["Frequency"] = np.log1p(features["Frequency"])
    features["Monetary"] = np.log1p(features["Monetary"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    best_k = find_best_k(X_scaled)
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    rfm["Cluster"] = km.fit_predict(X_scaled)

    return rfm

def label_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    summary = rfm.groupby("Cluster").agg(
        AvgRecency=("Recency", "mean"),
        AvgFrequency=("Frequency", "mean"),
        AvgMonetary=("Monetary", "mean"),
        CustomerCount=("CustomerID", "count"),
    ).round(1)

    # Rank clusters 
    summary["Score"] = (
        summary["AvgFrequency"].rank(ascending=True)
        + summary["AvgMonetary"].rank(ascending=True)
        - summary["AvgRecency"].rank(ascending=True)
    )
    ordered = summary.sort_values("Score", ascending=False).index.tolist()

    n = len(ordered)
    label_pool = ["Champions", "Loyal Customers", "Potential Loyalists",
                  "At Risk", "Needs Attention", "Hibernating", "Lost"]
    labels = label_pool[:n] if n <= len(label_pool) else \
        label_pool + [f"Segment {i}" for i in range(len(label_pool), n)]

    cluster_to_label = {cluster: labels[rank] for rank, cluster in enumerate(ordered)}
    rfm["Segment"] = rfm["Cluster"].map(cluster_to_label)

    recommendations = {
        "Champions": "Reward with early access & loyalty perks; ask for reviews/referrals.",
        "Loyal Customers": "Upsell higher-value products; enroll in a loyalty program.",
        "Potential Loyalists": "Offer targeted promotions to increase purchase frequency.",
        "At Risk": "Send personalized win-back offers before they churn completely.",
        "Needs Attention": "Re-engage with limited-time discounts and reminder emails.",
        "Hibernating": "Low-cost reactivation campaign; survey for feedback.",
        "Lost": "Deprioritize marketing spend; consider a final win-back attempt only.",
    }
    rfm["Recommendation"] = rfm["Segment"].map(recommendations)

    print("\n[Segments] Cluster profile summary:")
    display_summary = summary.copy()
    display_summary["Segment"] = [cluster_to_label[c] for c in display_summary.index]
    print(display_summary[["Segment", "CustomerCount", "AvgRecency", "AvgFrequency", "AvgMonetary"]])

    return rfm

# Exports
def make_segment_charts(rfm: pd.DataFrame):
    seg_counts = rfm["Segment"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    seg_counts.plot(kind="bar", ax=ax, color="#2E5EAA")
    ax.set_title("Customers per Segment")
    ax.set_ylabel("Customer Count")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/segment_counts.png")
    plt.close(fig)

    seg_revenue = rfm.groupby("Segment")["Monetary"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    seg_revenue.plot(kind="bar", ax=ax, color="#3E8E7E")
    ax.set_title("Total Revenue Contribution by Segment")
    ax.set_ylabel("Revenue")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/segment_revenue.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(rfm["Recency"], rfm["Frequency"], c=rfm["Cluster"], cmap="tab10", alpha=0.7)
    ax.set_xlabel("Recency (days since last purchase)")
    ax.set_ylabel("Frequency (# of orders)")
    ax.set_title("Customer Segments: Recency vs Frequency")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/segment_scatter.png")
    plt.close(fig)

    print(f"[Charts] Saved segment charts to {OUTPUT_DIR}/")


def main():
    df = load_and_clean(RAW_DATA_PATH)
    run_eda(df)
    rfm = build_rfm(df)
    rfm = run_clustering(rfm)
    rfm = label_segments(rfm)
    make_segment_charts(rfm)

    export_cols = ["CustomerID", "Recency", "Frequency", "Monetary", "Segment", "Recommendation"]
    rfm[export_cols].to_csv(f"{OUTPUT_DIR}/customer_segments_for_powerbi.csv", index=False)
    print(f"\n[Export] Power-BI-ready file written to {OUTPUT_DIR}/customer_segments_for_powerbi.csv")
    print("Import this CSV directly into Power BI Desktop (Get Data > Text/CSV) to build the dashboard.")


if __name__ == "__main__":
    main()
