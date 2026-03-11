#!/usr/bin/env python3
"""
P6 — Super-Eddington Quasar Sample Selection (SDSS Stripe 82)
Merkabit Prediction P6: alpha ~ 4/3 in structure functions near lambda_Edd ~ 1

Data: MacLeod et al. 2010, ApJ 721, 1014 — SDSS Stripe 82 quasar variability catalog
      VizieR: J/ApJ/721/1014
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import DATA_DIR, FIGURES_DIR, REPORTS_DIR

import warnings
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

QUASAR_DATA = os.path.join(DATA_DIR, 'quasars')
os.makedirs(QUASAR_DATA, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

VIZIER_URL = (
    "https://vizier.cds.unistra.fr/viz-bin/votable/-A?"
    "-source=J/ApJ/721/1014/table2&-out.max=unlimited"
)

# ─────────────────────────────────────────────────────────────────
# Fallback sample (representative quasars spanning lambda_Edd range)
# Used when catalog download is unavailable
# ─────────────────────────────────────────────────────────────────
FALLBACK_SAMPLE = [
    # (name, RA, Dec, z, Mi, logMBH, logLbol)
    ('SDSSJ000009.38+135618.4',  0.039,  13.938, 0.479, -22.65, 8.12, 44.85),
    ('SDSSJ000119.60+154543.2',  0.332,  15.762, 1.367, -25.89, 9.34, 46.42),
    ('SDSSJ000652.30+145856.1',  1.718,  14.982, 0.823, -23.78, 7.85, 45.34),
    ('SDSSJ001025.91+005447.6',  2.608,   0.913, 1.826, -27.12, 9.67, 47.01),
    ('SDSSJ001115.23+144601.8',  2.813,  14.767, 0.558, -23.12, 8.45, 45.12),
    ('SDSSJ001502.26+001212.4',  3.760,   0.203, 2.170, -26.45, 8.91, 46.78),
    ('SDSSJ002146.73-004847.1',  5.445,  -0.813, 1.092, -24.56, 8.23, 45.67),
    ('SDSSJ003043.59+000724.5',  7.682,   0.123, 0.295, -21.23, 7.56, 44.23),
    ('SDSSJ003421.22+152457.8',  8.588,  15.416, 1.543, -26.01, 9.12, 46.55),
    ('SDSSJ004054.65-091526.8', 10.228,  -9.257, 0.711, -23.45, 8.67, 45.23),
    ('SDSSJ005202.36+143221.1', 13.010,  14.539, 0.456, -22.34, 7.89, 44.67),
    ('SDSSJ010355.22+003125.9', 15.980,   0.524, 1.928, -27.56, 9.78, 47.23),
    ('SDSSJ011321.35+000735.4', 18.339,   0.126, 0.379, -21.89, 8.34, 44.56),
    ('SDSSJ012259.42-000807.6', 20.748,  -0.135, 0.632, -23.01, 7.45, 45.45),
    ('SDSSJ013435.67-090627.4', 23.649,  -9.108, 1.245, -25.23, 8.56, 46.12),
    ('SDSSJ014516.60+145510.6', 26.319,  14.920, 0.867, -24.12, 9.01, 45.89),
    ('SDSSJ015128.34+002433.1', 27.868,   0.409, 2.456, -28.01, 9.89, 47.45),
    ('SDSSJ020236.93+001635.1', 30.654,   0.276, 0.168, -20.56, 7.23, 43.89),
    ('SDSSJ021834.64+004715.2', 34.644,   0.788, 1.678, -26.78, 9.45, 46.89),
    ('SDSSJ023252.45-000826.3', 38.218,  -0.141, 0.523, -22.78, 8.78, 44.78),
]


def load_vizier_catalog():
    """Try to download the MacLeod et al. catalog from VizieR."""
    catalog_path = os.path.join(QUASAR_DATA, 'macleod2010_table2.csv')

    if os.path.exists(catalog_path):
        print(f"  Loading cached catalog: {catalog_path}")
        data = np.genfromtxt(catalog_path, delimiter=',', names=True,
                             dtype=None, encoding='utf-8')
        return data

    # Try astroquery
    try:
        from astroquery.vizier import Vizier
        v = Vizier(columns=['*'], row_limit=-1)
        result = v.get_catalogs('J/ApJ/721/1014/table2')
        if result:
            table = result[0]
            print(f"  Downloaded {len(table)} quasars from VizieR (astroquery)")
            table.write(catalog_path, format='csv', overwrite=True)
            return table
    except Exception as e:
        print(f"  astroquery unavailable: {e}")

    # Try direct HTTP
    try:
        import urllib.request
        votable_path = os.path.join(QUASAR_DATA, 'macleod2010.vot')
        print(f"  Downloading VOTable from VizieR...")
        urllib.request.urlretrieve(VIZIER_URL, votable_path)
        from astropy.io.votable import parse
        vot = parse(votable_path)
        table = vot.get_first_table().to_table()
        print(f"  Downloaded {len(table)} quasars from VizieR (VOTable)")
        table.write(catalog_path, format='csv', overwrite=True)
        return table
    except Exception as e:
        print(f"  Direct download failed: {e}")

    return None


def use_fallback():
    """Return fallback sample as parallel arrays."""
    print("  Using FALLBACK sample (20 representative quasars)")
    print("  *** Download full catalog for production analysis ***")
    names = [f[0] for f in FALLBACK_SAMPLE]
    ra = np.array([f[1] for f in FALLBACK_SAMPLE])
    dec = np.array([f[2] for f in FALLBACK_SAMPLE])
    z = np.array([f[3] for f in FALLBACK_SAMPLE])
    mi = np.array([f[4] for f in FALLBACK_SAMPLE])
    logmbh = np.array([f[5] for f in FALLBACK_SAMPLE])
    loglbol = np.array([f[6] for f in FALLBACK_SAMPLE])
    return names, ra, dec, z, mi, logmbh, loglbol


def compute_eddington_ratio(logmbh, loglbol):
    """Compute lambda_Edd = L_bol / L_Edd."""
    log_l_edd = 38.1 + logmbh  # L_Edd = 1.26e38 * M_BH/M_sun
    log_lambda_edd = loglbol - log_l_edd
    return 10.0**log_lambda_edd, log_lambda_edd


if __name__ == '__main__':
    print("=" * 70)
    print("P6 -- Super-Eddington Quasar Sample Selection")
    print("MacLeod et al. 2010, SDSS Stripe 82")
    print("=" * 70)

    catalog = load_vizier_catalog()

    if catalog is not None:
        try:
            cols = catalog.colnames if hasattr(catalog, 'colnames') else catalog.dtype.names
            name_key = 'SDSS' if 'SDSS' in cols else 'Name' if 'Name' in cols else None
            names_col = list(catalog[name_key]) if name_key else [f'Q{i}' for i in range(len(catalog))]
            ra = np.array(catalog['RAJ2000'], dtype=float)
            dec = np.array(catalog['DEJ2000'], dtype=float)
            z = np.array(catalog['z'], dtype=float)
            mi = np.array(catalog['Mi'], dtype=float)
            logmbh = np.array(catalog['logMBH'], dtype=float)
            loglbol = np.array(catalog['logLbol'], dtype=float)
            names = names_col
            is_fallback = False
            print(f"  Catalog loaded: {len(names)} quasars")
        except Exception as e:
            print(f"  Catalog parsing error: {e}")
            names, ra, dec, z, mi, logmbh, loglbol = use_fallback()
            is_fallback = True
    else:
        names, ra, dec, z, mi, logmbh, loglbol = use_fallback()
        is_fallback = True

    lambda_edd, log_lambda_edd = compute_eddington_ratio(logmbh, loglbol)

    # Select samples
    mask_valid = np.isfinite(log_lambda_edd) & np.isfinite(z) & np.isfinite(logmbh)
    idx_super = np.where(mask_valid & (lambda_edd > 1.0))[0]
    idx_near = np.where(mask_valid & (lambda_edd > 0.3) & (lambda_edd < 3.0))[0]
    idx_control = np.where(mask_valid & (lambda_edd < 0.1))[0]
    idx_all = np.where(mask_valid)[0]

    n_total = len(idx_all)
    n_super = len(idx_super)
    n_near = len(idx_near)
    n_control = len(idx_control)

    print(f"\n  Sample sizes:")
    print(f"    Total valid:         {n_total}")
    print(f"    Super-Edd (lam>1):   {n_super}")
    print(f"    Near-Edd (0.3-3):    {n_near}")
    print(f"    Control (lam<0.1):   {n_control}")

    # Save selected sample
    sample_path = os.path.join(QUASAR_DATA, 'super_edd_sample.csv')
    with open(sample_path, 'w') as f:
        f.write('name,ra,dec,z,Mi,logMBH,logLbol,lambda_edd,log_lambda_edd\n')
        for i in idx_near:
            f.write(f'{names[i]},{ra[i]:.6f},{dec[i]:.6f},{z[i]:.4f},'
                    f'{mi[i]:.2f},{logmbh[i]:.3f},{loglbol[i]:.3f},'
                    f'{lambda_edd[i]:.4f},{log_lambda_edd[i]:.4f}\n')
    print(f"\n  Saved near-Edd sample: {sample_path}")

    # Summary
    lines = []
    lines.append("=" * 70)
    lines.append("P6 -- Super-Eddington Quasar Sample Selection")
    lines.append("MacLeod et al. 2010, ApJ 721, 1014 -- SDSS Stripe 82")
    lines.append("=" * 70)
    lines.append(f"\nData source: {'FALLBACK (demo)' if is_fallback else 'VizieR J/ApJ/721/1014'}")
    lines.append(f"Total quasars: {n_total}")
    lines.append(f"\nSelection:")
    lines.append(f"  Super-Edd (lam > 1.0):  {n_super} ({100*n_super/max(n_total,1):.1f}%)")
    lines.append(f"  Near-Edd (0.3-3.0):     {n_near} ({100*n_near/max(n_total,1):.1f}%)")
    lines.append(f"  Control (lam < 0.1):    {n_control} ({100*n_control/max(n_total,1):.1f}%)")
    if n_near > 0:
        lines.append(f"\nNear-Edd statistics:")
        lines.append(f"  z median:       {np.median(z[idx_near]):.3f}")
        lines.append(f"  logMBH median:  {np.median(logmbh[idx_near]):.2f}")
        lines.append(f"  lambda median:  {np.median(lambda_edd[idx_near]):.3f}")
    lines.append(f"\nPrediction P6:")
    lines.append(f"  SF KWW exponent alpha should peak near 4/3 at lambda_Edd ~ 1.")
    lines.append(f"  Sub-Eddington control (lam << 1) should show alpha != 4/3.")
    lines.append(f"\nNext: run p6_quasar_kww.py for structure function analysis.")

    summary = '\n'.join(lines)
    print('\n' + summary)
    with open(os.path.join(REPORTS_DIR, 'p6_super_edd_summary.txt'), 'w') as f:
        f.write(summary)

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    ax = axes[0, 0]
    ax.hist(log_lambda_edd[idx_all], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', ls='--', lw=2, label='lam=1')
    ax.axvspan(np.log10(0.3), np.log10(3.0), alpha=0.1, color='green',
               label=f'Near-Edd ({n_near})')
    ax.set_xlabel('log10(lambda_Edd)')
    ax.set_ylabel('Count')
    ax.set_title('Eddington Ratio Distribution')
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    sc = ax.scatter(z[idx_all], log_lambda_edd[idx_all],
                    c=logmbh[idx_all], s=15, alpha=0.6, cmap='viridis')
    ax.axhline(0, color='red', ls='--', alpha=0.7)
    ax.set_xlabel('Redshift z')
    ax.set_ylabel('log10(lambda_Edd)')
    ax.set_title('Redshift vs Eddington Ratio')
    plt.colorbar(sc, ax=ax, label='log(M_BH)')

    ax = axes[1, 0]
    if n_control > 0:
        ax.scatter(z[idx_control], mi[idx_control], c='blue', s=10, alpha=0.3,
                   label=f'Sub-Edd ({n_control})')
    if n_super > 0:
        ax.scatter(z[idx_super], mi[idx_super], c='red', s=25, alpha=0.7,
                   label=f'Super-Edd ({n_super})')
    ax.set_xlabel('Redshift z')
    ax.set_ylabel('M_i')
    ax.set_title('Luminosity vs Redshift')
    ax.invert_yaxis()
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    ax.scatter(logmbh[idx_all], loglbol[idx_all], c=log_lambda_edd[idx_all],
               s=15, alpha=0.6, cmap='RdYlBu_r', vmin=-2, vmax=1)
    mbh_range = np.linspace(np.nanmin(logmbh[idx_all]), np.nanmax(logmbh[idx_all]), 50)
    ax.plot(mbh_range, 38.1 + mbh_range, 'r--', lw=2, label='L = L_Edd')
    ax.set_xlabel('log(M_BH / M_sun)')
    ax.set_ylabel('log(L_bol / erg/s)')
    ax.set_title('Black Hole Mass vs Luminosity')
    ax.legend(fontsize=9)

    tag = 'FALLBACK DEMO' if is_fallback else f'N = {n_total}'
    plt.suptitle(f'P6 -- SDSS Stripe 82 Quasar Sample ({tag})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'p6_super_edd_sample.png'), dpi=150)
    plt.close()
    print(f"\nSaved: p6_super_edd_sample.png")
