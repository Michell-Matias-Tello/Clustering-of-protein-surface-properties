import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_umap_clusters(X_umap, labels, title="UMAP Clusters"):
    """
    Plot UMAP clusters.
    """
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=X_umap[:, 0], y=X_umap[:, 1],
        hue=labels, palette="tab10", s=60, alpha=0.8
    )
    plt.title(title)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.savefig("results/figures/umap_clusters.png", dpi=300)
    plt.close()


def plot_cluster_profiles(cluster_means, title="Cluster Profiles"):
    """
    Plot cluster profiles as a heatmap.
    """
    plt.figure(figsize=(14, 6))
    sns.heatmap(cluster_means, annot=True, cmap="RdBu_r", center=0, fmt=".1f")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("results/figures/cluster_profiles_heatmap.png", dpi=300)
    plt.close()
