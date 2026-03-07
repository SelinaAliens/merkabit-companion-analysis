# Merkabit Framework: Stretched Exponential Analysis Across Quantum, Plasma, and Atmospheric Systems

Cross-platform analysis of stretched exponential (KWW) relaxation dynamics — from discrete time crystals to tokamak ELMs to hurricane rapid intensification — using the Merkabit framework.

## Key finding

The stretched exponential exponent alpha ~ 4/3 appears across multiple independent physical systems spanning 10 orders of magnitude in spatial scale, when relaxation is governed by cooperative threshold dynamics:

| Dataset | System | alpha | Platform | Scale |
|---------|--------|-------|----------|-------|
| Mi et al. 2022 | MBL DTC (strong disorder) | 0.822 | Google Sycamore | ~nm |
| Xiang et al. 2024 (high disorder) | Topological DTC + disorder | 0.72 | Google Sycamore | ~nm |
| Xiang et al. 2024 (zero disorder) | Topological DTC, clean | 1.34 | Google Sycamore | ~nm |
| Cavedon et al. 2019 — Idiv tail | ELM transport relaxation | 1.27 | ASDEX Upgrade tokamak | ~1 m |
| Cavedon et al. 2019 — gradTi | ELM ion temp. gradient recovery | 1.43 | ASDEX Upgrade tokamak | ~1 m |
| Hurricane Dorian 2019 — wind | Rapid intensification (TS to Cat 5) | 1.44 | NOAA recon | ~500 km |
| Hurricane Dorian 2019 — pressure | Pressure deepening (1003 to 910 mb) | 1.47 | NOAA recon | ~500 km |

The Merkabit signature (|alpha - 4/3| < 0.15, R^2 > 0.90) appears across quantum, plasma, and atmospheric systems. Spatial scale range: quantum dot (~10 nm) to hurricane (~500 km) = 10^10.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download Xiang 2024 and Hurricane Dorian datasets
python download_data.py
```

Mi 2022 and Randall 2021 data are included in the repository.

## Repository structure

```
merkabit-dtc/
├── config.py                # Path configuration (all scripts import from here)
├── download_data.py         # Fetches Xiang 2024 + Dorian data
├── requirements.txt
│
├── analysis/                # Core analysis scripts
│   ├── merkabit_analysis.py       # Mi et al. 2022 envelope fitting
│   ├── multi_dtc_analysis.py      # Cross-dataset comparison (Xiang + Randall)
│   ├── merkabit_refined.py        # Refined analysis variant
│   ├── subharmonic_envelope.py    # Subharmonic envelope extraction
│   ├── merkabit_predictions.py    # Testable predictions from the framework
│   ├── merkabit_predictions_final.py  # Final assessment with controls
│   ├── merkabit_predictions_caveats.py
│   ├── final_summary.py          # Cross-dataset summary report
│   ├── simulation.py             # DTC circuit simulator (pure-state + density matrix)
│   ├── kww_elm_final.py          # Cavedon 2019 ELM relaxation (ASDEX Upgrade)
│   ├── kww_tail_analysis.py       # Idiv tail sensitivity analysis
│   ├── kww_dorian_final.py        # Hurricane Dorian rapid intensification
│   ├── tsoukalas_alpha_verification.py
│   ├── nstar_stress_test.py
│   ├── short_time_scaling_test.py
│   └── r2_significance_test.py
│
├── validation/              # Supporting tests and stress tests
│   ├── stroboscopic_doubling_test.py   # Tests beta -> 2*beta hypothesis (RULED OUT)
│   ├── crossover_stress_test.py
│   ├── full_disorder_sweep.py
│   ├── haus_mode_locking_test.py
│   ├── q3q4_crossover_full_rebuild.py
│   └── ...
│
├── data/                    # Experimental datasets
│   ├── mi_2022/             # CSV files (included, 294 KB)
│   ├── randall_2021/        # JSON files (included, ~2.2 MB)
│   ├── xiang_2024/          # .mat files (downloaded via download_data.py)
│   ├── dorian/              # NetCDF files (downloaded via download_data.py)
│   └── README.md            # Data sources and DOIs
│
└── results/                 # Pre-computed figures and reports
    ├── figures/             # PNG analysis plots
    └── reports/             # Text summary reports
```

## Method

The subharmonic envelope is extracted from DTC autocorrelator data:

```
E(n) = (-1)^n * A(n)
```

where A(n) is the autocorrelator at Floquet cycle n. The envelope is fit to a stretched exponential:

```
A_env(n) = A0 * exp(-(n / n*)^alpha)
```

The exponent alpha characterises the decay dynamics:
- alpha < 1: stretched (sub-exponential), typical of MBL-stabilised DTCs
- alpha = 1: simple exponential
- alpha > 1: compressed (super-exponential), observed in clean exchange-coupled systems

## Datasets

See [data/README.md](data/README.md) for full details and Zenodo DOIs.

## References

- Mi et al., "Time-crystalline eigenstate order on a quantum processor", Nature 601, 531 (2022)
- Xiang et al., "Long-lived topological time-crystalline order on a quantum processor", Nat. Commun. 15, 8963 (2024)
- Randall et al., "Many-body-localized discrete time crystal with a programmable spin-based quantum simulator", Science 374, 1474 (2021)
- Cavedon et al., "Connecting the global H-mode power threshold to the local radial electric field at ASDEX Upgrade", Nucl. Fusion 60, 066026 (2020)
- NOAA Hurricane Research Division, Hurricane Dorian 2019 flight-level data, https://www.aoml.noaa.gov/hrd/Storm_pages/dorian2019/

## License

The analysis code in this repository is provided for research purposes. The experimental datasets are subject to their respective licenses as specified on Zenodo.
