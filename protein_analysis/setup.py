from setuptools import setup, find_packages

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
