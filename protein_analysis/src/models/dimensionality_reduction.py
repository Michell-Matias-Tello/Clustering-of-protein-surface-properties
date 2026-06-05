import umap
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
