import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import hdbscan


def find_optimal_clusters_umap(X_umap, max_k=10):
    """
    Find optimal number of clusters using silhouette score on UMAP-reduced data.
    """
    silhouette_scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_umap)
        score = silhouette_score(X_umap, labels)
        silhouette_scores.append(score)

    optimal_k = np.argmax(silhouette_scores) + 2
    return optimal_k


def cluster_data(X, n_clusters, algorithm="kmeans"):
    """
    Cluster data using the specified algorithm.
    """
    if algorithm == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    elif algorithm == "gmm":
        model = GaussianMixture(n_components=n_clusters, random_state=42)
    elif algorithm == "hierarchical":
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    elif algorithm == "dbscan":
        model = DBSCAN(eps=0.3, min_samples=5)
    elif algorithm == "hdbscan":
        model = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return model.fit_predict(X)
