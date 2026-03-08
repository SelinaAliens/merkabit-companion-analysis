# Data Sources

## Included in this repository

### Mi et al. 2022 — MBL DTC (Google Sycamore)
- **Directory**: `mi_2022/`
- **Paper**: Nature 601, 531 (2022)
- **DOI**: [10.5281/zenodo.5570676](https://doi.org/10.5281/zenodo.5570676)
- **Format**: CSV files from figures 2-5 and supplementary figures
- **Platform**: Google Sycamore, 20 transmon qubits

### Randall et al. 2021 — MBL DTC (Diamond NV centers)
- **Directory**: `randall_2021/`
- **Paper**: Science 374, 1474 (2021)
- **DOI**: [10.5281/zenodo.5636045](https://doi.org/10.5281/zenodo.5636045)
- **Format**: JSON files (correlation data, per-spin measurements, decay histograms)
- **Platform**: Diamond NV center, 9 spins

### Braboszcz et al. 2017 — Rishikesh Meditation Study (Phase 1 spectral data)
- **Directory**: `meditation/`
- **Paper**: Frontiers in Psychology 8, 1116 (2017)
- **DOI (spectral)**: [10.5281/zenodo.57911](https://doi.org/10.5281/zenodo.57911)
- **DOI (raw EEG)**: [10.5281/zenodo.2348892](https://doi.org/10.5281/zenodo.2348892)
- **Format**: MATLAB `.mat` files (spectral power, channel locations) + CSV summary tables
- **Groups**: CTR (0h), ISY/Shoonya (2,625h), VIP/Vipassana (9,201h), HYT (15,475h), 16 subjects each
- **Files included**: `711Hz_spec_data_medit.mat`, `60110Hz_spec_data_medit.mat`, `chanlocs.mat`, 4 CSV tables
- **Key result**: HYT alpha/gamma power ratio = 1.361, within 2.1% of 4/3; Z3 symmetry score HYT > CTR (p=0.037)
- **Deep dive findings**: ISY distribution unimodal (not bimodal), Moran's I = 0.091 (spatially clustered but sectorally uniform)
- **Windowed Z3 time series** (requires raw EEG): Sliding-window Z3 (30s window, 10s step) over full sessions. HYT > VIP (p=0.04), VIP suppressed (0% frac>1.0). P_MED_4 dynamic NOT confirmed (HYT not above CTR). Standout HYT_06 subject (median Z3=0.91, 31% above 1.0). Channel count confound noted (64ch vs 72ch affects H3/H2 normalization).
- **Phase 2** (not yet run): requires raw EEG (~6.3 GB from Zenodo 2348892) for KWW, |C|->1/3, Z3, Berry phase tests

## Downloaded via `download_data.py`

### Xiang et al. 2024 — Topological prethermal DTC (Google Sycamore)
- **Directory**: `xiang_2024/` (after running `python download_data.py`)
- **Paper**: Nat. Commun. 15, 8963 (2024)
- **DOI**: [10.5281/zenodo.13692134](https://doi.org/10.5281/zenodo.13692134)
- **Format**: MATLAB `.mat` files (main figures 2-4, supplementary figures)
- **Platform**: Google Sycamore, 18 superconducting qubits
- **Key file**: `main_figure2/fig2a.mat` — 6 qubits x 41 stroboscopic steps, alternating-sign DTC

### Cavedon et al. 2019 — ELM relaxation (ASDEX Upgrade tokamak)
- **File**: `figure2_embedded.png` (digitised from paper Figure 2)
- **Paper**: Nucl. Fusion 60, 066026 (2020)
- **Signals analysed**: Divertor current (Idiv) decay, ion temperature gradient (gradTi) recovery
- **Platform**: ASDEX Upgrade tokamak, shot #34430
- **Method**: Figure digitisation (pixel-to-physical calibration), KWW fitting
- **Key result**: Transport tail alpha = 1.27, gradTi recovery alpha = 1.43 — both within Merkabit window

### Hurricane Dorian 2019 — Rapid intensification (NOAA Hurricane Hunter)
- **Directory**: `dorian/` (after running `python download_data.py dorian`)
- **Source**: [NOAA HRD Flight-Level Data](https://www.aoml.noaa.gov/hrd/Storm_pages/dorian2019/)
- **Format**: NetCDF (`.nc`) flight-level data from Hurricane Hunter reconnaissance
- **Files**: `20190901H1_AC.nc` (52 MB), `20190901N1_AC.nc` (54 MB)
- **Storm**: Hurricane Dorian, Category 5, Sept 1 2019 — peak intensity 160 kt, 910 mb
- **Signals analysed**: Wind intensification time series (TS to Cat 5), pressure deepening, radial profiles
- **Key result**: Wind intensification alpha = 1.44, pressure deepening alpha = 1.47 — both within Merkabit window
- **Controls**: TS stage alpha = 3.0 (not 4/3), post-peak weakening alpha = 2.0 (different regime)
- **Spatial scale**: ~500 km (largest system in the Merkabit programme)

### CHB-MIT Scalp EEG — Epileptic seizure dynamics (PhysioNet)
- **Directory**: `chbmit/` (after running `python download_data.py chbmit`)
- **Source**: [PhysioNet CHB-MIT](https://physionet.org/content/chbmit/1.0.0/)
- **Paper**: Shoeb, A. H., "Application of machine learning to epileptic seizure onset detection and treatment", PhD Thesis, MIT (2009)
- **Format**: EDF files (scalp EEG, 23 channels, 256 Hz)
- **Patient**: chb01, 7 seizures across 7 EDF files + 2 seizure-free files for interictal baseline
- **Signals analysed**: KWW envelope ACF (pre-ictal, near-onset, ictal, interictal windows), Riemannian covariance geometry
- **Key result**: Peak pre-ictal alpha = 1.23 (mean), 5/7 seizures within Merkabit window; alpha rises toward 4/3 as seizure approaches
- **Controls**: Interictal alpha = 0.79 (below 1.036 in all 7 cases), ictal alpha = 0.68
- **Significance**: First biological system in the Merkabit table; spatial scale ~10 cm (cortical)

## Datasets analysed but without public data

- **Shinjo et al. 2026** — Clean 2D DTC (IBM Heron, 133 qubits). GitHub repo is private.
- **Kyprianidis et al. 2021** — Prethermal DTC (Trapped ions). No public data archive.
