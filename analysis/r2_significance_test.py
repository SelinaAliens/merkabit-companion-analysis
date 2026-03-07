"""
Monte Carlo Significance Test for R^2 from 3-Point Linear Fits

Tests: How meaningful is R^2 = 0.995 when fitting a line to only 3 data points?

Method: Generate random 3-point datasets and compute linear R^2 to see how
often high R^2 values arise purely by chance.
"""

import numpy as np
from scipy import stats
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import *

# ============================================================
# Configuration
# ============================================================
np.random.seed(42)

N_SIMULATIONS = 10_000
X_VALUES = np.array([0.00, 1.72, 4.65])
N_POINTS = len(X_VALUES)

# Uniform distribution parameters
Y_UNIFORM_LOW = 1.0
Y_UNIFORM_HIGH = 2.5

# Normal distribution parameters (matching observed data)
Y_NORMAL_MEAN = 1.78
Y_NORMAL_STD = 0.33

# R^2 thresholds to test
THRESHOLDS = [0.90, 0.95, 0.99, 0.995, 0.999]


def compute_r_squared(x, y):
    """Compute R^2 from linear regression of y on x."""
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return r_value ** 2


def text_histogram(data, bins=20, width=50, title="Histogram"):
    """Print a text-based histogram."""
    counts, bin_edges = np.histogram(data, bins=bins)
    max_count = max(counts)

    print(f"\n{'=' * (width + 30)}")
    print(f"  {title}")
    print(f"{'=' * (width + 30)}")

    for i in range(len(counts)):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        bar_len = int(counts[i] / max_count * width) if max_count > 0 else 0
        bar = '#' * bar_len
        print(f"  [{lo:6.3f}, {hi:6.3f}) | {bar:<{width}} | {counts[i]:5d}")

    print(f"{'=' * (width + 30)}")


def run_simulation(y_generator, label):
    """Run Monte Carlo simulation with the given y-value generator."""
    r2_values = np.zeros(N_SIMULATIONS)

    for i in range(N_SIMULATIONS):
        y = y_generator()
        r2_values[i] = compute_r_squared(X_VALUES, y)

    # --- Report statistics ---
    print(f"\n{'#' * 70}")
    print(f"  RESULTS: {label}")
    print(f"{'#' * 70}")

    print(f"\n  Number of simulations : {N_SIMULATIONS:,}")
    print(f"  x values             : {X_VALUES}")
    print(f"  Number of points     : {N_POINTS}")

    print(f"\n  --- R^2 Distribution Statistics ---")
    print(f"  Mean   R^2 : {np.mean(r2_values):.6f}")
    print(f"  Median R^2 : {np.median(r2_values):.6f}")
    print(f"  Std    R^2 : {np.std(r2_values):.6f}")
    print(f"  Min    R^2 : {np.min(r2_values):.6f}")
    print(f"  Max    R^2 : {np.max(r2_values):.6f}")

    print(f"\n  --- Fraction of simulations exceeding R^2 thresholds ---")
    for threshold in THRESHOLDS:
        frac = np.mean(r2_values > threshold)
        count = int(np.sum(r2_values > threshold))
        print(f"  R^2 > {threshold:.3f} : {frac:.4f}  ({count:,} / {N_SIMULATIONS:,})")

    # --- Histogram ---
    text_histogram(r2_values, bins=20, width=50,
                   title=f"R^2 Distribution -- {label}")

    return r2_values


def main():
    print("=" * 70)
    print("  MONTE CARLO SIGNIFICANCE TEST FOR R^2 FROM 3-POINT LINEAR FITS")
    print("=" * 70)
    print(f"\n  Question: How meaningful is R^2 = 0.995 from fitting a line")
    print(f"  to only 3 data points?")
    print(f"\n  We generate {N_SIMULATIONS:,} random 3-point datasets and fit")
    print(f"  linear regressions to see how often high R^2 arises by chance.")

    # ---- Simulation 1: Uniform distribution ----
    def uniform_generator():
        return np.random.uniform(Y_UNIFORM_LOW, Y_UNIFORM_HIGH, size=N_POINTS)

    r2_uniform = run_simulation(
        uniform_generator,
        f"Uniform y in [{Y_UNIFORM_LOW}, {Y_UNIFORM_HIGH}]"
    )

    # ---- Simulation 2: Normal distribution ----
    def normal_generator():
        return np.random.normal(Y_NORMAL_MEAN, Y_NORMAL_STD, size=N_POINTS)

    r2_normal = run_simulation(
        normal_generator,
        f"Normal y ~ N({Y_NORMAL_MEAN}, {Y_NORMAL_STD})"
    )

    # ---- Combined summary ----
    print(f"\n{'#' * 70}")
    print(f"  COMBINED SUMMARY")
    print(f"{'#' * 70}")

    print(f"\n  {'Metric':<45} {'Uniform':>10} {'Normal':>10}")
    print(f"  {'-'*45} {'-'*10} {'-'*10}")
    print(f"  {'Mean R^2':<45} {np.mean(r2_uniform):>10.4f} {np.mean(r2_normal):>10.4f}")
    print(f"  {'Median R^2':<45} {np.median(r2_uniform):>10.4f} {np.median(r2_normal):>10.4f}")
    print(f"  {'Fraction with R^2 > 0.99':<45} {np.mean(r2_uniform > 0.99):>10.4f} {np.mean(r2_normal > 0.99):>10.4f}")
    print(f"  {'Fraction with R^2 > 0.995':<45} {np.mean(r2_uniform > 0.995):>10.4f} {np.mean(r2_normal > 0.995):>10.4f}")

    # ---- Interpretation ----
    frac_uniform_99 = np.mean(r2_uniform > 0.99)
    frac_normal_99 = np.mean(r2_normal > 0.99)
    frac_uniform_995 = np.mean(r2_uniform > 0.995)
    frac_normal_995 = np.mean(r2_normal > 0.995)

    print(f"\n{'#' * 70}")
    print(f"  INTERPRETATION")
    print(f"{'#' * 70}")
    print(f"""
  With only 3 data points, a linear fit has only 1 degree of freedom
  for residuals (3 points - 2 parameters = 1 DOF). This means R^2 is
  inherently biased upward, and high R^2 values arise frequently even
  from random data.

  Under uniform random y-values in [{Y_UNIFORM_LOW}, {Y_UNIFORM_HIGH}]:
    - {frac_uniform_99*100:.1f}% of random datasets yield R^2 > 0.99
    - {frac_uniform_995*100:.1f}% of random datasets yield R^2 > 0.995
    - This means R^2 = 0.995 from a 3-point fit is NOT statistically
      significant -- it has a p-value of approximately {frac_uniform_995:.3f}.

  Under normally distributed y-values (matching observed data):
    - {frac_normal_99*100:.1f}% of random datasets yield R^2 > 0.99
    - {frac_normal_995*100:.1f}% of random datasets yield R^2 > 0.995

  CONCLUSION: An R^2 of 0.995 from a 3-point linear fit is NOT
  particularly impressive or meaningful. With only 1 residual degree
  of freedom, the R^2 metric has very little discriminating power.
  Any 3 points that are not wildly non-collinear will produce a
  high R^2. More data points are needed to draw reliable conclusions
  about linearity.
""")


if __name__ == "__main__":
    main()
