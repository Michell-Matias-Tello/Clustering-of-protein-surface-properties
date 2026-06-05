"""
Script to generate a complete project structure for a Protein Analysis Project.
Run this script in the root directory to create directories, files, and Jupyter notebooks.
"""

import os
import json
from pathlib import Path

# Define the project structure and file contents
PROJECT_STRUCTURE = {
    "data/raw/.gitkeep": "",
    "data/processed/.gitkeep": "",
    "data/README.md": "# Data Directory\n\n- `raw/`: Store raw data files.\n- `processed/`: Store processed data files.\n",
    
    "notebooks/exploratory/.gitkeep": "",
    "notebooks/modeling/.gitkeep": "",
    "notebooks/visualization/.gitkeep": "",
    
    "src/data/__init__.py": "",
    "src/data/data_generator.py": '''import numpy as np
import pandas as pd


def generate_synthetic_protein_data(n_samples=500, random_seed=42):
    """
    Generate synthetic protein dataset with realistic distributions for clustering analysis.
    """
    np.random.seed(random_seed)

    area_superficial = np.random.lognormal(mean=9.5, sigma=0.4, size=n_samples)
    area_superficial = np.clip(area_superficial, 5000, 35000).astype(int)

    carga_neta = np.random.normal(loc=-2.0, scale=6.0, size=n_samples)
    carga_neta = np.clip(carga_neta, -25, 25).round(1)

    solubilidad = np.random.gamma(shape=2.0, scale=10.0, size=n_samples)
    solubilidad = np.clip(solubilidad, 0.5, 120).round(1)

    tendencia_agreg = np.random.beta(a=1.5, b=5.0, size=n_samples)
    tendencia_agreg = np.clip(tendencia_agreg, 0, 1).round(3)

    zeta = np.random.normal(loc=-15.0, scale=10.0, size=n_samples)
    zeta = np.clip(zeta, -40, 10).round(1)

    enturbiamiento = np.random.normal(loc=60.0, scale=12.0, size=n_samples)
    enturbiamiento = np.clip(enturbiamiento, 35, 85).round(1)

    puentes_salinos = np.random.poisson(lam=3.0, size=n_samples)
    puentes_salinos = np.clip(puentes_salinos, 0, 15).astype(int)

    accesibilidad_trp = np.random.uniform(0, 100, size=n_samples).round(1)

    rugosidad = np.random.lognormal(mean=0.35, sigma=0.2, size=n_samples)
    rugosidad = np.clip(rugosidad, 1.0, 10.0).round(2)

    data = pd.DataFrame({
        'accessible_surface_area': area_superficial,
        'net_surface_charge': carga_neta,
        'solubility': solubilidad,
        'aggregation_tendency': tendencia_agreg,
        'zeta_potential': zeta,
        'clouding_point': enturbiamiento,
        'salt_bridges': puentes_salinos,
        'tryptophan_accessibility': accesibilidad_trp,
        'surface_roughness': rugosidad
    })

    return data


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    df = generate_synthetic_protein_data()
    output_path = Path(__file__).parent.parent.parent / "data" / "raw" / "synthetic_protein_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Synthetic protein data generated and saved to {output_path}")
''',
    "src/data/data_loader.py": '''import pandas as pd
from pathlib import Path


def load_data(file_path):
    """
    Load protein data from a CSV file.
    """
    return pd.read_csv(Path(file_path))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    data = load_data("data/raw/synthetic_protein_data.csv")
    print(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
''',
    
    "src/models/__init__.py": "",
    "src/models/dimensionality_reduction.py": '''import umap
import numpy as np
from sklearn.preprocessing import StandardScaler


def reduce_dimensions(data, n_components=2, random_seed=42):
    """
    Reduce dimensionality using UMAP.
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    reducer = umap.UMAP(n_components=n_components, random_state=random_seed)
    return reducer.fit_transform(data_scaled)
''',
    "src/models/clustering.py": '''import numpy as np
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
''',
    
    "src/visualization/__init__.py": "",
    "src/visualization/plot_clusters.py": '''import matplotlib
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
''',
    
    "src/__init__.py": "",
    "src/main.py": '''import sys
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
''',
    
    "tests/__init__.py": "",
    "tests/test_data_generator.py": '''import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import unittest
import pandas as pd
from data.data_generator import generate_synthetic_protein_data


class TestDataGenerator(unittest.TestCase):
    def test_generate_synthetic_protein_data(self):
        df = generate_synthetic_protein_data(n_samples=100, random_seed=42)
        self.assertEqual(df.shape[0], 100)
        self.assertEqual(df.shape[1], 9)


if __name__ == '__main__':
    unittest.main()
''',
    "tests/test_clustering.py": '''import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import unittest
import numpy as np
from models.clustering import find_optimal_clusters_umap, cluster_data


class TestClustering(unittest.TestCase):
    def test_find_optimal_clusters_umap(self):
        X_umap = np.random.rand(100, 2)
        optimal_k = find_optimal_clusters_umap(X_umap, max_k=5)
        self.assertTrue(2 <= optimal_k <= 5)

    def test_cluster_data(self):
        X = np.random.rand(100, 2)
        labels = cluster_data(X, n_clusters=3, algorithm="kmeans")
        self.assertEqual(len(np.unique(labels)), 3)


if __name__ == '__main__':
    unittest.main()
''',
    
    "config/params.yaml": '''data:
  n_samples: 500
  random_seed: 42

model:
  n_components: 2
  max_k: 10
  algorithm: kmeans

paths:
  raw_data: data/raw/synthetic_protein_data.csv
  processed_data: data/processed/features.csv
  results: results/
''',
    "config/paths.yaml": '''raw_data_dir: data/raw/
processed_data_dir: data/processed/
results_dir: results/
figures_dir: results/figures/
models_dir: results/models/
''',
    
    "setup.py": '''from setuptools import setup, find_packages

setup(
    name="protein_analysis",
    version="0.1.0",
    description="A Python project for clustering and analyzing synthetic protein data.",
    author="Michell Matias Tello",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "umap-learn>=0.5.0",
        "hdbscan>=0.8.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
    ],
    python_requires=">=3.8",
)
''',
    
    ".gitignore": '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/
.venv/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Data
data/raw/
data/processed/
results/

# Jupyter
.ipynb_checkpoints

# OS
.DS_Store
Thumbs.db
''',
    
    "README.md": '''# Protein Analysis Project

A Python project for clustering and analyzing synthetic protein data using UMAP and various clustering algorithms.

## Project Structure

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



## Setup

1. Clone the repository:
   git clone https://github.com/yourusername/protein_analysis.git
   cd protein_analysis

2. Create a virtual environment:
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate

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
''',
}

# Define a function to create a basic Jupyter notebook
def create_jupyter_notebook(title, code_cells):
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"# {title}\n", f"This notebook covers {title.lower()}.\n"]
            }
        ] + [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [cell_code]
            }
            for cell_code in code_cells
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    return json.dumps(notebook, indent=2)


# Define notebook contents
NOTEBOOKS = {
    "notebooks/exploratory/01_initial_exploration.ipynb": create_jupyter_notebook(
        "Initial Data Exploration",
        [
            "import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns",
            "# Load data\ndf = pd.read_csv('../../data/raw/synthetic_protein_data.csv')\nprint(df.head())",
            "# Basic statistics\nprint(df.describe())",
            "# Plot distributions\ndf.hist(bins=20, figsize=(12, 10))\nplt.tight_layout()\nplt.show()"
        ]
    ),
    "notebooks/modeling/02_clustering_analysis.ipynb": create_jupyter_notebook(
        "Clustering Analysis",
        [
            "import pandas as pd\nimport numpy as np\nfrom sklearn.cluster import KMeans\nfrom sklearn.preprocessing import StandardScaler\nimport umap",
            "# Load data\ndf = pd.read_csv('../../data/raw/synthetic_protein_data.csv')\nX = df.values",
            "# Standardize data\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)",
            "# UMAP reduction\nreducer = umap.UMAP(n_components=2, random_state=42)\nX_umap = reducer.fit_transform(X_scaled)",
            "# KMeans clustering\nkmeans = KMeans(n_clusters=3, random_state=42, n_init=10)\nlabels = kmeans.fit_predict(X_umap)\ndf['cluster'] = labels\nprint(df['cluster'].value_counts())"
        ]
    ),
    "notebooks/visualization/03_cluster_visualization.ipynb": create_jupyter_notebook(
        "Cluster Visualization",
        [
            "import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns",
            "# Load clustered data\ndf = pd.read_csv('../../data/processed/clustered_data.csv')",
            "# Plot UMAP clusters\nplt.figure(figsize=(10, 8))\nsns.scatterplot(data=df, x='accessible_surface_area', y='solubility', hue='cluster', palette='tab10')\nplt.title('Clusters by Surface Area and Solubility')\nplt.show()",
            "# Heatmap of cluster means\ncluster_means = df.groupby('cluster').mean()\nplt.figure(figsize=(12, 6))\nsns.heatmap(cluster_means, annot=True, cmap='coolwarm')\nplt.title('Cluster Profiles')\nplt.show()"
        ]
    )
}

# Merge notebooks into PROJECT_STRUCTURE
PROJECT_STRUCTURE.update(NOTEBOOKS)


def create_project_structure(base_dir="protein_analysis"):
    """
    Create the project structure and files in the specified base directory.
    """
    base_path = Path(base_dir)
    
    for file_path in PROJECT_STRUCTURE.keys():
        full_path = base_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
    
    for file_path, content in PROJECT_STRUCTURE.items():
        full_path = base_path / file_path
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
    
    print(f"Project structure created at: {base_path.absolute()}")


if __name__ == "__main__":
    create_project_structure()
    print("To run the project, use: python -m src.main")