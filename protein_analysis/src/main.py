import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data.data_generator import generate_synthetic_protein_data
from data.data_loader import load_data
from models.dimensionality_reduction import reduce_dimensions
from models.clustering import find_optimal_clusters_umap, cluster_data
from visualization.plot_clusters import plot_umap_clusters, plot_cluster_profiles
import os


def main():
    base_dir = Path(__file__).parent.parent
    os.chdir(base_dir)
    
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/processed", exist_ok=True)

    df = generate_synthetic_protein_data()
    X = df.values
    X_umap = reduce_dimensions(X)
    optimal_k = find_optimal_clusters_umap(X_umap)
    print(f"Optimal number of clusters: {optimal_k}")
    labels = cluster_data(X_umap, n_clusters=optimal_k, algorithm="kmeans")
    df["cluster"] = labels
    plot_umap_clusters(X_umap, labels, title=f"UMAP Clusters (k={optimal_k})")
    cluster_means = df.groupby("cluster").mean()
    plot_cluster_profiles(cluster_means, title=f"Cluster Profiles (k={optimal_k})")
    df.to_csv("data/processed/clustered_data.csv", index=False)
    cluster_means.to_csv("results/processed/cluster_means.csv")
    print("Analysis complete. Results saved to data/processed/ and results/processed/")


if __name__ == "__main__":
    main()
