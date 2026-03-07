"""
Path configuration for Merkabit DTC analysis scripts.
All paths are relative to the repository root.
"""
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(REPO_ROOT, 'data')
MI_2022_DATA = os.path.join(DATA_DIR, 'mi_2022')
XIANG_2024_DATA = os.path.join(DATA_DIR, 'xiang_2024')
RANDALL_2021_DATA = os.path.join(DATA_DIR, 'randall_2021')
DORIAN_DATA = os.path.join(DATA_DIR, 'dorian')

# Xiang 2024 subdirectory (created by download_data.py)
XIANG_2024_BASE = os.path.join(
    XIANG_2024_DATA,
    'xlelephant-Long-lived-topological-time-crystalline-order-on-a-quantum-processor-168eca4'
)

# Results directories
RESULTS_DIR = os.path.join(REPO_ROOT, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
REPORTS_DIR = os.path.join(RESULTS_DIR, 'reports')

# Create output directories if they don't exist
for d in [FIGURES_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)
