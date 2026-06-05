# ============================================================
# 1. IMPORTS
# ============================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score

import umap
import hdbscan

# ============================================================
# 2. FOLDER STRUCTURE SETUP
# ============================================================
# Create the necessary folders for the project
os.makedirs('data', exist_ok=True)
os.makedirs('output/data', exist_ok=True)
os.makedirs('output/visualizations', exist_ok=True)

# ============================================================
# 3. LOAD DATA
# ============================================================
df = pd.read_csv('data/protein_surface_data.csv')
print("Data shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())

# ============================================================
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("\n=== Descriptive Statistics ===")
print(df.describe().round(2))

print("\n=== Missing values ===")
print(df.isnull().sum())

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Protein Surface Properties')
plt.tight_layout()
plt.savefig('output/visualizations/correlation_heatmap.png')
plt.close()
print("\nCorrelation heatmap saved as 'output/visualizations/correlation_heatmap.png'")

# ============================================================
# 5. PREPROCESSING – SCALING
# ============================================================
X = df.values
print("\nOriginal feature matrix shape:", X.shape)

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
print("Scaled matrix shape:", X_scaled.shape)
print("\nFirst 2 rows after scaling (rounded):")
print(np.round(X_scaled[:2], 2))

# ============================================================
# 6. PCA FOR VARIANCE EXPLORATION (not used for clustering)
# ============================================================
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_
cumulative = explained_variance.cumsum()

print("\nExplained variance ratio per component:")
for i, ev in enumerate(explained_variance[:5], 1):
    print(f"PC{i}: {ev:.3f}")
print(f"\nCumulative variance for first 4 components: {cumulative[3]:.3f}")

# Plot cumulative variance
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative) + 1), cumulative, 'bo-')
plt.axhline(y=0.80, color='r', linestyle='--', label='80% threshold')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA – Cumulative Variance')
plt.legend()
plt.grid(True)
plt.savefig('output/visualizations/pca_elbow.png')
plt.close()
print("PCA elbow plot saved as 'output/visualizations/pca_elbow.png'")

# ============================================================
# 7. NON‑LINEAR DIMENSIONALITY REDUCTION (UMAP)
# ============================================================
reducer = umap.UMAP(n_components=5, random_state=42)
X_umap = reducer.fit_transform(X_scaled)
print("\nUMAP reduced shape:", X_umap.shape)

# ============================================================
# 8. DETERMINE OPTIMAL k (ELBOW + SILHOUETTE) ON UMAP SPACE
# ============================================================
inertias_umap = []
silhouettes_umap = []
k_range = range(2, 9)

print("\n--- K-means on UMAP space ---")
for k in k_range:
    kmeans_umap = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_umap = kmeans_umap.fit_predict(X_umap)
    inertias_umap.append(kmeans_umap.inertia_)
    sil = silhouette_score(X_umap, labels_umap)
    silhouettes_umap.append(sil)
    print(f"k={k}: inertia={kmeans_umap.inertia_:.0f}, silhouette={sil:.3f}")

# Plot elbow and silhouette
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(k_range, inertias_umap, 'o-', color='green')
ax1.set_xlabel('Number of clusters (k)')
ax1.set_ylabel('Inertia')
ax1.set_title('UMAP + K-means – Elbow Method')
ax2.plot(k_range, silhouettes_umap, 'o-', color='green')
ax2.set_xlabel('Number of clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('UMAP + K-means – Silhouette Score')
plt.tight_layout()
plt.savefig('output/visualizations/kmeans_umap_elbow_silhouette.png')
plt.close()
print("UMAP version plot saved as 'output/visualizations/kmeans_umap_elbow_silhouette.png'")

# ============================================================
# 9. FINAL K-MEANS MODEL (k=8)
# ============================================================
optimal_k = 8
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_umap)
df['cluster_umap'] = cluster_labels

print("\nCluster label counts (UMAP + K-means):")
print(df['cluster_umap'].value_counts().sort_index())

sil = silhouette_score(X_umap, cluster_labels)
db = davies_bouldin_score(X_umap, cluster_labels)
print(f"\nSilhouette score for k={optimal_k} on UMAP space: {sil:.3f}")
print(f"Davies-Bouldin index on UMAP space: {db:.3f}")

# ============================================================
# 10. VISUALIZATION – PCA PROJECTION (coloured by clusters)
# ============================================================
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7, edgecolors='k')
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)')
plt.title(f'Protein Clusters (UMAP + K-means, k={optimal_k}) – PCA Projection')
plt.colorbar(scatter, label='Cluster')
plt.savefig('output/visualizations/pca_clusters_umap.png')
plt.close()
print("PCA cluster plot saved as 'output/visualizations/pca_clusters_umap.png'")

# ============================================================
# 11. CLUSTER PROFILING (MEAN VALUES PER CLUSTER)
# ============================================================
cluster_means = df.groupby('cluster_umap').mean(numeric_only=True)
print("\n=== Average feature values per cluster (original units) ===")
print(cluster_means.round(2))

# Heatmap of cluster means
plt.figure(figsize=(14, 6))
sns.heatmap(cluster_means, annot=True, cmap='RdBu_r', center=0, fmt='.1f')
plt.title(f'Cluster Profiles (k={optimal_k}) – Mean Values per Feature')
plt.tight_layout()
plt.savefig('output/visualizations/cluster_profiles_heatmap_umap.png')
plt.close()
print("Heatmap saved as 'output/visualizations/cluster_profiles_heatmap_umap.png'")

# ============================================================
# 12. ALTERNATIVE ALGORITHMS (COMPARISON)
# ============================================================
print("\n--- Alternative Algorithms on UMAP space ---")

# GMM (k=3)
gmm = GaussianMixture(n_components=3, random_state=42)
labels_gmm = gmm.fit_predict(X_umap)
sil_gmm = silhouette_score(X_umap, labels_gmm)
print(f"GMM k=3: silhouette={sil_gmm:.3f}")

# Hierarchical (k=3, ward)
hier = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_hier = hier.fit_predict(X_umap)
sil_hier = silhouette_score(X_umap, labels_hier)
print(f"Hierarchical k=3: silhouette={sil_hier:.3f}")

# DBSCAN (eps=0.3)
db = DBSCAN(eps=0.3, min_samples=5)
labels_db = db.fit_predict(X_umap)
n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
noise_db = sum(labels_db == -1)
if n_clusters_db > 1:
    sil_db = silhouette_score(X_umap[labels_db != -1], labels_db[labels_db != -1])
    print(f"DBSCAN: clusters={n_clusters_db}, noise={noise_db}, silhouette={sil_db:.3f}")

# HDBSCAN
hdb = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
labels_hdb = hdb.fit_predict(X_umap)
n_clusters_hdb = len(set(labels_hdb)) - (1 if -1 in labels_hdb else 0)
noise_hdb = sum(labels_hdb == -1)
if n_clusters_hdb > 1:
    sil_hdb = silhouette_score(X_umap[labels_hdb != -1], labels_hdb[labels_hdb != -1])
    print(f"HDBSCAN: clusters={n_clusters_hdb}, noise={noise_hdb}, silhouette={sil_hdb:.3f}")

# ============================================================
# 13. SAVE DATA WITH CLUSTERS
# ============================================================
df.to_csv('output/data/protein_surface_data_with_clusters.csv', index=False)
print("\nData with clusters saved as 'output/data/protein_surface_data_with_clusters.csv'")

# ============================================================
# 14. FINAL MESSAGE
# ============================================================
print("\n--- PROCESS COMPLETED ---")
print("All visualization and data files have been saved in their respective folders.")