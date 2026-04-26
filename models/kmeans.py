"""
models/kmeans.py
----------------
K-Means Clustering for Used Car Market Segmentation.

The model groups cars into price segments:
    Cluster 0 → Budget
    Cluster 1 → Mid-Range
    Cluster 2 → Premium
    (labels assigned after training based on cluster centres)

Functions:
    train_kmeans(X)            -> model
    predict_cluster(model, X)  -> list[str]
    get_cluster_label(cluster_id, model) -> str
"""

import numpy as np
from sklearn.cluster import KMeans


# Human-readable segment names (assigned after training by price order)
SEGMENT_NAMES = ["Budget", "Mid-Range", "Premium"]


def train_kmeans(X, n_clusters: int = 3, random_state: int = 42):
    """
    Train a K-Means clustering model on the given feature matrix.

    Parameters
    ----------
    X          : array-like, shape (n_samples, n_features) — scaled features
    n_clusters : number of clusters / market segments
    random_state : seed for reproducibility

    Returns
    -------
    Fitted KMeans model
    """
    model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",      # smarter initialisation → faster convergence
        n_init=10,
        max_iter=300,
        random_state=random_state,
    )
    model.fit(X)
    print(f"[K-Means] Training complete. Clusters: {n_clusters}  |  Inertia: {model.inertia_:.2f}")
    return model


def predict_cluster(model, X) -> list:
    """
    Predict the cluster index for each row in X.

    Parameters
    ----------
    model : fitted KMeans model
    X     : array-like, shape (n_samples, n_features) — scaled features

    Returns
    -------
    list of integer cluster indices
    """
    return model.predict(X).tolist()


def get_segment_map(model, y_original) -> dict:
    """
    Map raw cluster indices → human-readable segment names.

    Strategy: sort clusters by their mean Selling_Price; the cheapest
    cluster → 'Budget', middle → 'Mid-Range', most expensive → 'Premium'.

    Parameters
    ----------
    model      : fitted KMeans model
    y_original : original (unscaled) Selling_Price Series / array

    Returns
    -------
    dict {cluster_index: segment_name}
    """
    y_arr = np.array(y_original)
    cluster_ids = model.labels_
    n_clusters = model.n_clusters

    # Compute mean price per cluster
    mean_prices = {
        cid: y_arr[cluster_ids == cid].mean()
        for cid in range(n_clusters)
    }

    # Sort clusters cheapest → most expensive
    sorted_clusters = sorted(mean_prices, key=mean_prices.get)

    # Assign labels (pad or trim if n_clusters ≠ 3)
    labels = SEGMENT_NAMES[:n_clusters] + [f"Segment {i}" for i in range(n_clusters - len(SEGMENT_NAMES))]
    return {cid: labels[rank] for rank, cid in enumerate(sorted_clusters)}


def get_cluster_label(cluster_id: int, segment_map: dict) -> str:
    """Return the human-readable segment name for a raw cluster index."""
    return segment_map.get(cluster_id, f"Cluster {cluster_id}")
