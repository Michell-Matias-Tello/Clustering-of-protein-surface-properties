import pandas as pd
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
