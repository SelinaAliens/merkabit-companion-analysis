"""
Download datasets for the Merkabit DTC analysis.

Usage:
    python download_data.py          # Download all datasets
    python download_data.py xiang    # Download only Xiang 2024
    python download_data.py chbmit   # Download only CHB-MIT EEG
    python download_data.py meditation  # Download meditation raw EEG
"""
import os
import sys
import zipfile
import urllib.request
import shutil

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_ROOT, 'data')

DATASETS = {
    'xiang': {
        'name': 'Xiang et al. 2024 — Topological prethermal DTC',
        'doi': '10.5281/zenodo.13692134',
        'url': 'https://zenodo.org/records/13692134/files/xlelephant-Long-lived-topological-time-crystalline-order-on-a-quantum-processor-168eca4.zip',
        'dest': os.path.join(DATA_DIR, 'xiang_2024'),
        'size_mb': 3.1,
    },
    'dorian': {
        'name': 'Hurricane Dorian 2019 — NOAA Hurricane Hunter flight-level data',
        'doi': 'NOAA HRD',
        'url': 'https://www.aoml.noaa.gov/hrd/Storm_pages/dorian2019/flight_level/20190901H1_AC.nc',
        'url2': 'https://www.aoml.noaa.gov/hrd/Storm_pages/dorian2019/flight_level/20190901N1_AC.nc',
        'dest': os.path.join(DATA_DIR, 'dorian'),
        'size_mb': 106,
    },
    'chbmit': {
        'name': 'CHB-MIT Scalp EEG — Epileptic seizure dynamics (PhysioNet)',
        'doi': 'PhysioNet',
        'url_base': 'https://physionet.org/files/chbmit/1.0.0/chb01/',
        'files': [
            'chb01_01.edf', 'chb01_03.edf', 'chb01_04.edf', 'chb01_09.edf',
            'chb01_15.edf', 'chb01_16.edf', 'chb01_18.edf', 'chb01_21.edf',
            'chb01_26.edf',
        ],
        'dest': os.path.join(DATA_DIR, 'chbmit'),
        'size_mb': 450,
    },
    'meditation': {
        'name': 'Rishikesh Meditation Raw EEG — Braboszcz et al. 2017 (Phase 2)',
        'doi': '10.5281/zenodo.2348892',
        'url': 'https://zenodo.org/records/2348892/files/Meditation_EEG_raw.zip',
        'dest': os.path.join(DATA_DIR, 'meditation', 'raw_eeg'),
        'size_mb': 6300,
    },
}

# Mi 2022 and Randall 2021 data are small enough to be included in the repo directly.
# Cavedon 2019 data is digitised from the published figure (included as figure2_embedded.png).
# CHB-MIT data is downloaded per-file from PhysioNet (only chb01 patient files needed).
# Meditation Phase 1 spectral data (.mat + CSV) is included in data/meditation/ directly.
# Meditation Phase 2 raw EEG (~6.3 GB) must be downloaded separately.
# See data/README.md for all sources.


def download_file(url, dest_path):
    """Download a file with progress reporting."""
    print(f'  Downloading: {url}')
    print(f'  Destination: {dest_path}')

    def reporthook(count, block_size, total_size):
        if total_size > 0:
            pct = min(100, count * block_size * 100 // total_size)
            print(f'\r  Progress: {pct}%', end='', flush=True)

    urllib.request.urlretrieve(url, dest_path, reporthook)
    print()


def download_dataset(key):
    ds = DATASETS[key]
    print(f'\n{"="*60}')
    print(f'Dataset: {ds["name"]}')
    print(f'DOI: {ds["doi"]}')
    print(f'Size: ~{ds["size_mb"]} MB')
    print(f'{"="*60}')

    dest = ds['dest']
    os.makedirs(dest, exist_ok=True)

    zip_path = os.path.join(dest, f'{key}.zip')

    # Check if already downloaded
    contents = os.listdir(dest)
    if any(d.endswith('.nc') for d in contents) or any(d.startswith('xlelephant') for d in contents):
        print(f'  Already downloaded. Skipping.')
        return
    if any(d.endswith('.edf') for d in contents):
        print(f'  Already downloaded. Skipping.')
        return

    # Handle CHB-MIT: multiple individual EDF files from PhysioNet
    if 'files' in ds:
        url_base = ds['url_base']
        for fname in ds['files']:
            fpath = os.path.join(dest, fname)
            if os.path.exists(fpath):
                print(f'  {fname} already exists, skipping.')
                continue
            download_file(url_base + fname, fpath)
        print(f'  Done.')
        return

    # Handle datasets with multiple files (e.g., Dorian NetCDF)
    urls = [ds['url']]
    if 'url2' in ds:
        urls.append(ds['url2'])

    for url in urls:
        fname = url.split('/')[-1]
        fpath = os.path.join(dest, fname)
        if url.endswith('.nc'):
            download_file(url, fpath)
        else:
            download_file(url, zip_path)
            print(f'  Extracting...')
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(dest)
            os.remove(zip_path)

    print(f'  Done.')


def main():
    if len(sys.argv) > 1:
        keys = sys.argv[1:]
    else:
        keys = list(DATASETS.keys())

    for key in keys:
        if key not in DATASETS:
            print(f'Unknown dataset: {key}')
            print(f'Available: {", ".join(DATASETS.keys())}')
            sys.exit(1)
        download_dataset(key)

    print(f'\nAll downloads complete.')
    print(f'Mi 2022 and Randall 2021 data are included in the repo (data/mi_2022/, data/randall_2021/).')


if __name__ == '__main__':
    main()
