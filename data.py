import os
import numpy as np
import pandas as pd

# ============================================================
# FOLDER STRUCTURE SETUP
# ============================================================
# Create the necessary folders for the project
os.makedirs('data', exist_ok=True)
os.makedirs('output/data', exist_ok=True)
os.makedirs('output/visualizations', exist_ok=True)

# Set a seed for reproducible results
np.random.seed(42)

n_rows = 500

# ============================================================
# SYNTHETIC DATA GENERATION
# ============================================================

# 1. Accessible surface area (Å²) - log-normal distribution
accessible_surface_area = np.random.lognormal(mean=9.5, sigma=0.4, size=n_rows)
accessible_surface_area = np.clip(accessible_surface_area, 5000, 35000).astype(int)

# 2. Net surface charge (real value) - normal distribution
net_surface_charge = np.random.normal(loc=-2.0, scale=6.0, size=n_rows)
net_surface_charge = np.clip(net_surface_charge, -25, 25).round(1)

# 3. Estimated solubility (mg/mL) - gamma distribution
estimated_solubility = np.random.gamma(shape=2.0, scale=10.0, size=n_rows)
estimated_solubility = np.clip(estimated_solubility, 0.5, 120).round(1)

# 4. Aggregation tendency (0-1) - beta distribution
aggregation_tendency = np.random.beta(a=1.5, b=5.0, size=n_rows)
aggregation_tendency = np.clip(aggregation_tendency, 0, 1).round(3)

# 5. Simulated zeta potential (mV) - negative normal distribution
zeta_potential = np.random.normal(loc=-15.0, scale=10.0, size=n_rows)
zeta_potential = np.clip(zeta_potential, -40, 10).round(1)

# 6. Clouding (turbidity) point (°C) - normal distribution
clouding_point = np.random.normal(loc=60.0, scale=12.0, size=n_rows)
clouding_point = np.clip(clouding_point, 35, 85).round(1)

# 7. Number of salt bridges (integer) - Poisson distribution
salt_bridges = np.random.poisson(lam=3.0, size=n_rows)
salt_bridges = np.clip(salt_bridges, 0, 15).astype(int)

# 8. Tryptophan residue accessibility (%) - uniform distribution
tryptophan_accessibility = np.random.uniform(0, 100, size=n_rows).round(1)

# 9. Surface roughness - log-normal distribution
surface_roughness = np.random.lognormal(mean=0.35, sigma=0.2, size=n_rows)
surface_roughness = np.clip(surface_roughness, 1.0, 2.5).round(2)

# 10. Average hydration (water molecules/residue) - gamma distribution
average_hydration = np.random.gamma(shape=3.0, scale=0.8, size=n_rows)
average_hydration = np.clip(average_hydration, 2.0, 8.0).round(1)

# ============================================================
# DATAFRAME CREATION
# ============================================================
df = pd.DataFrame({
    'accessible_surface_area': accessible_surface_area,
    'net_surface_charge': net_surface_charge,
    'estimated_solubility': estimated_solubility,
    'aggregation_tendency': aggregation_tendency,
    'zeta_potential': zeta_potential,
    'clouding_point': clouding_point,
    'salt_bridges': salt_bridges,
    'tryptophan_accessibility': tryptophan_accessibility,
    'surface_roughness': surface_roughness,
    'average_hydration': average_hydration
})

# ============================================================
# SAVE DATA TO CSV
# ============================================================
df.to_csv('data/protein_surface_data.csv', index=False)

# ============================================================
# VALIDATION AND FINAL MESSAGE
# ============================================================
print("--- DATA GENERATED ---")
print(df.head())
print(f"\nFile 'data/protein_surface_data.csv' generated with {n_rows} rows.")
print("\nFolder structure created:")
print("  - data/")
print("  - output/data/")
print("  - output/visualizations/")