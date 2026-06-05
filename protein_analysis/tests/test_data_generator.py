import sys
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
