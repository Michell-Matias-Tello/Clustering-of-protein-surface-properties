# Protein Analysis Project

A Python project for clustering and analyzing synthetic protein data using UMAP and various clustering algorithms.

## Project Structure
````
protein_surface_clustering/
│
├── data/
│   ├── protein_surface_data.csv
│   └── protein_analysis/
│       ├── raw/
│       │   └── synthetic_protein_data.csv
│       └── processed/
│           └── clustered_data.csv
│
├── output/
│   ├── data/
│   │   └── protein_surface_data_with_clusters.csv
│   └── visualizations/
│       ├── correlation_heatmap.png
│       ├── pca_elbow.png
│       ├── kmeans_umap_elbow_silhouette.png
│       ├── pca_clusters_umap.png
│       └── cluster_profiles_heatmap_umap.png
│
├── protein_analysis/
│   ├── notebooks/
│   │   ├── exploratory/
│   │   │   └── 01_initial_exploration.ipynb
│   │   ├── modeling/
│   │   │   └── 02_clustering_analysis.ipynb
│   │   └── visualization/
│   │       └── 03_cluster_visualization.ipynb
│   │
│   ├── src/
│   │   ├── data/
│   │   │   ├── data_generator.py
│   │   │   └── data_loader.py
│   │   ├── models/
│   │   │   ├── clustering.py
│   │   │   └── dimensionality_reduction.py
│   │   ├── visualization/
│   │   │   ├── plot_clusters.py
│   │   │   └── plot_features.py
│   │   └── main.py
│   │
│   ├── tests/
│   │   ├── test_data_generator.py
│   │   └── test_clustering.py
│   │
│   ├── results/
│   │   ├── figures/
│   │   │   ├── cluster_profiles_heatmap.png
│   │   │   └── umap_clusters.png
│   │   └── processed/
│   │       ├── clustered_data.csv
│   │       └── cluster_means.csv
│   │
│   ├── config/
│   │   ├── params.yaml
│   │   └── paths.yaml
│   │
│   ├── .gitignore
│   ├── README.md
│   ├── requirements.txt
│   └── setup.py
│
├── data.py
├── structure.py
├── main.py
├── README.md
└── requirements.txt

````

## Setup

1. Clone the repository:
   git clone https://github.com/yourusername/protein_analysis.git
   cd protein_analysis

2. Create a virtual environment:
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt

## Usage

Run the main analysis:
python src/main.py

## Features

- Synthetic protein data generation
- Dimensionality reduction with UMAP
- Clustering with KMeans, GMM, Hierarchical, DBSCAN, HDBSCAN
- Visualization of clusters and profiles
- Modular and reusable code structure

## License

This project is intended for portfolio and educational purposes.
