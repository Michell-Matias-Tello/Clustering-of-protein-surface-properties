import sys
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
