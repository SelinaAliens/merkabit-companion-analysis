# Merkabit Framework: Stretched Exponential Analysis Across Quantum, Plasma, Atmospheric, Biological, and Neural Systems

Cross-platform analysis of stretched exponential (KWW) relaxation dynamics — from discrete time crystals to tokamak ELMs to hurricane rapid intensification to epileptic seizures to meditation EEG — using the Merkabit framework.

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
| CHB-MIT chb01 — peak pre-ictal | EEG seizure cooperative cascade | 1.23 | PhysioNet EEG | ~10 cm |
| Rishikesh Meditation — HYT | Alpha/gamma spectral ratio | 1.36 | Scalp EEG 64ch | ~10 cm |

The Merkabit signature (|alpha - 4/3| < 0.15, R^2 > 0.90) appears across quantum, plasma, atmospheric, and biological systems. Spatial scale range: quantum dot (~10 nm) to hurricane (~500 km) = 10^10. The meditation analysis extends the biological domain with spectral balance approaching 4/3 in experienced practitioners.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download Xiang 2024, Dorian, CHB-MIT, and meditation raw EEG datasets
python download_data.py
```

Mi 2022, Randall 2021, and meditation Phase 1 spectral data are included in the repository.

## Repository structure

```
merkabit-dtc/
├── config.py                # Path configuration (all scripts import from here)
├── download_data.py         # Fetches Xiang 2024, Dorian, CHB-MIT, + meditation raw EEG
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
│   ├── chbmit_analysis.py         # CHB-MIT EEG seizure analysis (9th dataset)
│   ├── meditation_phase1_analysis.py    # Rishikesh meditation Phase 1 spectral
│   ├── phase1_channel_analysis.py       # 64-channel topographic alpha/gamma
│   ├── phase1_deep_analysis.py          # ISY distribution, Z3 symmetry, spatial uniformity
│   ├── rishikesh_meditation_phase2.py   # Phase 2: KWW + geometric (requires raw EEG)
│   ├── z3_timeseries.py                # Windowed Z3 time series (sliding-window H3/H2)
│   ├── delorme_mindwandering_phase2.py  # Within-subject MW vs MED analysis
│   ├── tsoukalas_alpha_verification.py
│   ├── nstar_stress_test.py
│   ├── short_time_scaling_test.py
│   ├── r2_significance_test.py
│   ├── tesseract_zero_point_investigation.py   # Zero point constant: self-sustaining coherence
│   ├── tesseract_zero_point_part2.py           # Deep analysis: Berry phases, 24-cell, winding
│   ├── tesseract_zero_point_part3.py           # Spectral gap ladder: 1/n, (n-1)/n, 1/3 → 4/3
│   ├── twentyfour_cell_investigation.py        # 24-cell: dual ouroboros, 600-cell falsification
│   └── tesseract_merkabit_simulation.py        # 4×4 counter-rotating tesseract simulation
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
│   ├── chbmit/              # EDF files (downloaded via download_data.py)
│   ├── meditation/          # MAT + CSV spectral data (included, ~62 KB)
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

## Theoretical foundations

The zero point constant (1/3) and KWW exponent (4/3) are derived from pure graph theory with zero free parameters:

- **24-cell spectral gap/bandwidth = 4/12 = 1/3** — unique among all regular 4-polytopes (600-cell falsification: gap/bw = 0.146, not 1/3)
- **KWW exponent = (1/3)/(1/4) = 4/3** — spectral enhancement ratio from tesseract to 24-cell
- **Dimensional ladder**: n-cube gap/bw = 1/n (exact), n-orthoplex gap/bw = (n-1)/n (complementary)
- **Self-sustaining coherence**: 4-spinor |C|_mean = 0.47, 8-spinor |C|_mean = 0.33 ≈ 1/3 under internal dynamics alone
- **Dual ouroboros Berry phase = exact zero**: 12 forward + 12 inverse steps → perfect state closure

See `analysis/tesseract_zero_point_investigation.py` (Parts 1-3) and `analysis/twentyfour_cell_investigation.py`.

## Datasets

See [data/README.md](data/README.md) for full details and Zenodo DOIs.

## References

- Mi et al., "Time-crystalline eigenstate order on a quantum processor", Nature 601, 531 (2022)
- Xiang et al., "Long-lived topological time-crystalline order on a quantum processor", Nat. Commun. 15, 8963 (2024)
- Randall et al., "Many-body-localized discrete time crystal with a programmable spin-based quantum simulator", Science 374, 1474 (2021)
- Cavedon et al., "Connecting the global H-mode power threshold to the local radial electric field at ASDEX Upgrade", Nucl. Fusion 60, 066026 (2020)
- NOAA Hurricane Research Division, Hurricane Dorian 2019 flight-level data, https://www.aoml.noaa.gov/hrd/Storm_pages/dorian2019/
- Shoeb, A. H. "Application of machine learning to epileptic seizure onset detection and treatment", PhD Thesis, MIT (2009). CHB-MIT dataset: https://physionet.org/content/chbmit/1.0.0/
- Braboszcz et al., "Plasticity of visual attention in Isha yoga meditation practitioners", Frontiers in Psychology 8, 1116 (2017). Zenodo: https://doi.org/10.5281/zenodo.57911 (spectral), https://doi.org/10.5281/zenodo.2348892 (raw EEG)

## License

The analysis code in this repository is provided for research purposes. The experimental datasets are subject to their respective licenses as specified on Zenodo.
