import numpy as np
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
