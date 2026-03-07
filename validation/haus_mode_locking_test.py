"""
Haus Master Equation: Mode-Locking Threshold Exponent Test
===========================================================

HYPOTHESIS: The KWW exponent at the mode-locking threshold for N modes
equals log_{N-1}(N). For N=12: alpha = log_11(12) = ln(12)/ln(11) ~ 1.0363.

If confirmed, this would establish log_11(12) as a universal constant of
standing wave formation, derivable from the mathematics of phase
synchronisation itself.

APPROACH:
  Model 1: Kuramoto model (phase dynamics, pairwise coupling)
  Model 2: Haus-like four-wave-mixing model (amplitude+phase, cubic coupling)
  Model 3: Cooperative cascade model (sequential mode-locking with feedback)
  Model 4: Analytical multi-exponential eigenvalue spectrum

For each model:
  1. Find the mode-locking threshold K_c for N modes
  2. Start from locked state at K slightly below K_c
  3. Track coherence decay r(T)
  4. Fit KWW: r(T) ~ exp(-(T/tau)^alpha)
  5. Extract alpha at threshold
  6. Sweep N = 4,6,8,10,12,14,16,20
  7. Compare alpha(N) vs log_{N-1}(N)

Reference: Haus, H.A. (2000) IEEE JSTQE 6, 1173.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit, brentq
import warnings
import sys
import os

warnings.filterwarnings('ignore')
np.random.seed(42)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import *

# ==============================================================================
# UTILITIES
# ==============================================================================

def kww(t, tau, alpha, A0):
    """KWW stretched/compressed exponential"""
    return A0 * np.exp(-(t / tau) ** alpha)


def fit_kww(t, r, alpha_bounds=(0.3, 4.0)):
    """
    Fit KWW to coherence decay. Returns (tau, alpha, A0, R2).
    Requires r to be monotonically decreasing from ~1 to ~0.
    """
    # Filter valid data
    mask = np.isfinite(r) & (r > 0.01) & (r < r[0] * 1.1)
    t_f, r_f = t[mask], r[mask]
    if len(t_f) < 5:
        return np.nan, np.nan, np.nan, 0.0

    try:
        # Estimate tau from half-life
        half_idx = np.argmin(np.abs(r_f - r_f[0] * np.exp(-1)))
        tau_est = max(t_f[half_idx], t_f[1])
        p0 = [tau_est, 1.0, r_f[0]]
        bounds = ([t_f[1] * 0.1, alpha_bounds[0], 0.5 * r_f[0]],
                  [t_f[-1] * 5, alpha_bounds[1], 1.5 * r_f[0]])
        popt, _ = curve_fit(kww, t_f, r_f, p0=p0, bounds=bounds, maxfev=20000)
        ss_res = np.sum((r_f - kww(t_f, *popt)) ** 2)
        ss_tot = np.sum((r_f - np.mean(r_f)) ** 2)
        R2 = 1 - ss_res / max(ss_tot, 1e-15)
        return popt[0], popt[1], popt[2], R2
    except Exception:
        return np.nan, np.nan, np.nan, 0.0


def log_prediction(N):
    """The predicted threshold exponent: log_{N-1}(N)"""
    return np.log(N) / np.log(N - 1)


# ==============================================================================
# MODEL 1: KURAMOTO MODEL
# ==============================================================================

def kuramoto_rhs(t, theta, N, K, omega):
    """Kuramoto model: dtheta_i/dt = omega_i + (K/N) * sum_j sin(theta_j - theta_i)"""
    sin_diff = np.sin(theta[np.newaxis, :] - theta[:, np.newaxis])
    coupling = np.sum(sin_diff, axis=1)
    return omega + (K / N) * coupling


def kuramoto_order_param(theta):
    """Order parameter r = |<exp(i*theta)>|"""
    return np.abs(np.mean(np.exp(1j * theta)))


def find_kuramoto_locked_state(N, K, omega, max_iter=5000):
    """Find the locked state of the Kuramoto model via relaxation."""
    theta = np.zeros(N)
    dt = 0.01
    for _ in range(max_iter):
        dth = kuramoto_rhs(0, theta, N, K, omega)
        # Subtract mean to remove overall rotation
        dth -= np.mean(dth)
        theta += dt * dth
        if np.max(np.abs(dth)) < 1e-10:
            break
    return theta


def kuramoto_threshold(N, dw=1.0):
    """
    Find the critical coupling K_c for N equally-spaced oscillators.
    Locked state exists when all phases satisfy |theta_n - mean| < pi/2.
    For equally spaced omega_n = (n - (N-1)/2)*dw, the approximate threshold is
    K_c ~ (N-1)*dw/pi (from the condition omega_max/(K*r) < 1).
    We find it precisely by binary search on stability.
    """
    omega = (np.arange(N) - (N - 1) / 2.0) * dw

    def is_locked(K):
        if K < 0.1:
            return False
        theta_locked = find_kuramoto_locked_state(N, K, omega)
        r = kuramoto_order_param(theta_locked)
        if r < 0.5:
            return False
        # Check stability: perturb and see if it returns
        theta_pert = theta_locked + 0.01 * np.random.randn(N)
        sol = solve_ivp(kuramoto_rhs, [0, 50], theta_pert,
                        args=(N, K, omega), method='RK45',
                        rtol=1e-8, atol=1e-10)
        if not sol.success:
            return False
        r_final = kuramoto_order_param(sol.y[:, -1])
        return r_final > 0.8

    # Binary search
    K_low, K_high = 0.1, N * dw
    for _ in range(50):
        K_mid = (K_low + K_high) / 2
        if is_locked(K_mid):
            K_high = K_mid
        else:
            K_low = K_mid
    return (K_low + K_high) / 2


def run_kuramoto_decay(N, K, omega, T_max=300, n_trials=50):
    """
    Start from locked state, evolve at coupling K (below threshold).
    Track coherence decay r(T).
    """
    # Find locked state at K*1.5 (well above threshold)
    theta_locked = find_kuramoto_locked_state(N, K * 1.5, omega)

    dt_save = T_max / 500
    t_eval = np.arange(0, T_max, dt_save)
    r_trials = []

    for trial in range(n_trials):
        theta0 = theta_locked + 0.02 * np.random.randn(N)
        try:
            sol = solve_ivp(kuramoto_rhs, [0, T_max], theta0,
                            args=(N, K, omega), method='RK45',
                            t_eval=t_eval, rtol=1e-8, atol=1e-10)
            if sol.success:
                r = np.array([kuramoto_order_param(sol.y[:, i])
                              for i in range(len(sol.t))])
                r_trials.append(r)
        except Exception:
            continue

    if len(r_trials) < 5:
        return t_eval, np.full_like(t_eval, np.nan), np.full_like(t_eval, np.nan)

    r_arr = np.array(r_trials)
    return t_eval, np.mean(r_arr, axis=0), np.std(r_arr, axis=0)


# ==============================================================================
# MODEL 2: HAUS-LIKE FOUR-WAVE-MIXING MODEL
# ==============================================================================

def haus_rhs(t, y, N, params):
    """
    Haus master equation for N modes in frequency domain.
    y = [Re(a_0), Im(a_0), ..., Re(a_{N-1}), Im(a_{N-1})]

    da_n/dT = [g_n - l] * a_n + gamma * [|A|^2 * A]_n

    g_n = g0 * (1 - (n_eff * dw_wg)^2)  parabolic gain
    gamma = saturable absorber coefficient
    Four-wave mixing computed via FFT.
    """
    g0, l0, dw_wg, gamma = params

    # Extract complex amplitudes
    a = y[::2] + 1j * y[1::2]
    n_eff = np.arange(N) - (N - 1) / 2.0

    # Gain profile
    g = g0 * (1.0 - (n_eff * dw_wg) ** 2)

    # Four-wave mixing via FFT
    Nfft = max(4 * N, 64)
    # Build A(t) using IFFT
    spec = np.zeros(Nfft, dtype=complex)
    for i in range(N):
        idx = int(n_eff[i]) % Nfft
        spec[idx] = a[i]
    A_t = np.fft.ifft(spec) * Nfft  # scale for unnormalized

    # |A|^2 * A in time domain
    IA = np.abs(A_t) ** 2 * A_t

    # Back to modes
    IA_spec = np.fft.fft(IA) / Nfft
    fwm = np.zeros(N, dtype=complex)
    for i in range(N):
        idx = int(n_eff[i]) % Nfft
        fwm[i] = IA_spec[idx]

    # Mode equations
    da = (g - l0) * a + gamma * fwm

    # Pack back
    dydt = np.empty(2 * N)
    dydt[::2] = da.real
    dydt[1::2] = da.imag
    return dydt


def haus_order_param(y, N):
    """Phase coherence for Haus model modes."""
    a = y[::2] + 1j * y[1::2]
    norms = np.abs(a)
    norms = np.maximum(norms, 1e-15)
    phases = np.angle(a)
    # Weight by amplitude
    weights = norms / np.sum(norms)
    r = np.abs(np.sum(weights * np.exp(1j * phases)))
    return r


def find_haus_threshold(N, params_base, gamma_range=(0.001, 0.5)):
    """Find mode-locking threshold for the Haus model by binary search."""
    g0, l0, dw_wg, _ = params_base

    def is_locked(gamma):
        params = (g0, l0, dw_wg, gamma)
        # Start from locked state
        n_eff = np.arange(N) - (N - 1) / 2.0
        y0 = np.zeros(2 * N)
        for i in range(N):
            amp = max(0.1, 1.0 - 0.3 * (n_eff[i] * dw_wg) ** 2)
            y0[2 * i] = amp  # all in phase
        try:
            sol = solve_ivp(haus_rhs, [0, 200], y0, args=(N, params),
                            method='RK45', rtol=1e-8, atol=1e-10, max_step=2.0)
            if not sol.success:
                return False
            r_final = haus_order_param(sol.y[:, -1], N)
            return r_final > 0.8
        except Exception:
            return False

    # Binary search
    g_low, g_high = gamma_range
    for _ in range(40):
        g_mid = (g_low + g_high) / 2
        if is_locked(g_mid):
            g_high = g_mid
        else:
            g_low = g_mid
    return (g_low + g_high) / 2


def run_haus_decay(N, gamma, params_base, T_max=300, n_trials=30):
    """
    Start from locked state, evolve at gamma (below threshold).
    Track coherence decay.
    """
    g0, l0, dw_wg, _ = params_base
    params = (g0, l0, dw_wg, gamma)
    n_eff = np.arange(N) - (N - 1) / 2.0

    dt_save = T_max / 400
    t_eval = np.arange(0, T_max, dt_save)
    r_trials = []

    for trial in range(n_trials):
        y0 = np.zeros(2 * N)
        for i in range(N):
            amp = max(0.1, 1.0 - 0.3 * (n_eff[i] * dw_wg) ** 2)
            phase = 0.02 * np.random.randn()
            y0[2 * i] = amp * np.cos(phase)
            y0[2 * i + 1] = amp * np.sin(phase)
        try:
            sol = solve_ivp(haus_rhs, [0, T_max], y0, args=(N, params),
                            method='RK45', t_eval=t_eval,
                            rtol=1e-8, atol=1e-10, max_step=2.0)
            if sol.success:
                r = np.array([haus_order_param(sol.y[:, i], N)
                              for i in range(len(sol.t))])
                r_trials.append(r)
        except Exception:
            continue

    if len(r_trials) < 5:
        return t_eval, np.full_like(t_eval, np.nan), np.full_like(t_eval, np.nan)

    r_arr = np.array(r_trials)
    return t_eval, np.mean(r_arr, axis=0), np.std(r_arr, axis=0)


# ==============================================================================
# MODEL 3: COOPERATIVE CASCADE
# ==============================================================================

def cooperative_cascade_decay(N, K_rel, T_max=20, n_trials=500):
    """
    Cooperative cascade: N modes, each can unlock with rate that increases
    as more modes unlock. K_rel = K/K_c (relative coupling).

    At K_rel < 1: locked state is unstable. Unlocking cascades.
    Rate of k-th mode unlocking (given k-1 already unlocked):
      lambda_k = (1 - K_rel) + K_rel * (k-1)/N

    i.e., base instability rate (1 - K_rel) enhanced by the fraction of
    modes already unlocked. This is the cooperative feedback.

    The coherence is r(t) = (N - n_unlocked(t)) / N.
    """
    dt = 0.01
    n_steps = int(T_max / dt)
    t_arr = np.arange(n_steps) * dt
    r_trials = []

    for _ in range(n_trials):
        n_unlocked = 0
        r_trace = np.zeros(n_steps)
        for step in range(n_steps):
            r_trace[step] = (N - n_unlocked) / N
            if n_unlocked >= N:
                r_trace[step:] = 0
                break
            # Rate for next unlocking
            lam = max(0, (1 - K_rel) + K_rel * n_unlocked / N)
            # Poisson probability of unlocking in dt
            p_unlock = 1 - np.exp(-lam * dt)
            if np.random.rand() < p_unlock:
                n_unlocked += 1
        r_trials.append(r_trace)

    r_arr = np.array(r_trials)
    return t_arr, np.mean(r_arr, axis=0), np.std(r_arr, axis=0)


# ==============================================================================
# MODEL 4: ANALYTICAL MULTI-EXPONENTIAL
# ==============================================================================

def analytical_multiexp_alpha(N, eigenvalue_type='kuramoto'):
    """
    For N modes with a specific eigenvalue spectrum, compute the
    multi-exponential decay and fit KWW to get alpha.

    The coherence decay is: r(t) = (1/N) * sum_{k=1}^{N-1} exp(-lambda_k * t)

    At threshold, the slowest eigenvalue lambda_1 -> 0.
    We set lambda_1 = epsilon (small) and compute the effective alpha.
    """
    # Eigenvalue spectrum
    if eigenvalue_type == 'kuramoto':
        # Near threshold, the Kuramoto phase stability eigenvalues for
        # equally spaced frequencies scale as:
        # lambda_k ~ K_c * (1 - cos(2*pi*k/N)) / 2 - |omega_k|
        # At threshold, lambda_1 = 0, lambda_k > 0 for k > 1
        # We parametrise as lambda_k = sin^2(pi*k/(2*N)) for k=1,...,N-1
        # normalised so lambda_1 = sin^2(pi/(2N)) ~ (pi/(2N))^2 ~ 0
        k = np.arange(1, N)
        lam = np.sin(np.pi * k / (2 * N)) ** 2
    elif eigenvalue_type == 'linear':
        # Linear spectrum: lambda_k = k/(N-1)
        k = np.arange(1, N)
        lam = k / (N - 1)
    elif eigenvalue_type == 'quadratic':
        # Quadratic: lambda_k = (k/(N-1))^2
        k = np.arange(1, N)
        lam = (k / (N - 1)) ** 2
    else:
        raise ValueError(f"Unknown eigenvalue type: {eigenvalue_type}")

    # Set slowest mode to a small epsilon (at threshold)
    # Scale so that lambda_1 = epsilon, others proportionally
    epsilon = 0.001
    lam = lam * (epsilon / lam[0])

    # Compute the multi-exponential decay
    t_max = 5.0 / epsilon  # enough for full decay
    t = np.linspace(0, t_max, 2000)
    r = np.zeros_like(t)
    for k_idx in range(len(lam)):
        r += np.exp(-lam[k_idx] * t) / (N - 1)

    # Fit KWW
    tau, alpha, A0, R2 = fit_kww(t, r)
    return alpha, R2, lam


# ==============================================================================
# MODEL 5: GENERALIZED COOPERATIVE CASCADE (continuous)
# ==============================================================================

def cooperative_ode_decay(N, K_rel, T_max=50, n_points=2000):
    """
    Continuous cooperative cascade model.

    The number of locked modes m(t) satisfies:
      dm/dt = -rate(m) where rate(m) = (1 - K_rel)*m + K_rel * m * (N - m) / N

    i.e., modes unlock at a rate proportional to:
      - the base instability (1 - K_rel) * (number still locked)
      - enhanced by cooperative feedback: coupling to already-unlocked modes

    Coherence: r(t) = m(t) / N

    This is a mean-field ODE version of Model 3.
    """
    def rhs(t, m):
        if m <= 0:
            return 0.0
        n_unlocked = N - m
        # Unlocking rate per locked mode
        rate_per = abs(1 - K_rel) + K_rel * n_unlocked / N
        return -rate_per * m

    m0 = N - 0.01  # start nearly fully locked
    t_span = (0, T_max)
    t_eval = np.linspace(0, T_max, n_points)

    sol = solve_ivp(rhs, t_span, [m0], t_eval=t_eval, method='RK45',
                    rtol=1e-10, atol=1e-12)
    if sol.success:
        r = np.clip(sol.y[0] / N, 0, 1)
        return sol.t, r
    else:
        return t_eval, np.full(n_points, np.nan)


# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================

def run_full_analysis():
    print("=" * 78)
    print("HAUS MODE-LOCKING THRESHOLD EXPONENT TEST")
    print("Hypothesis: alpha_threshold = log_{N-1}(N) for N cooperative modes")
    print("For N=12: alpha = log_11(12) = ln(12)/ln(11) = %.6f" % log_prediction(12))
    print("=" * 78)

    # N values to test
    N_values = [4, 6, 8, 10, 12, 14, 16, 20]
    dw = 1.0  # frequency spacing

    results = {
        'N': [], 'prediction': [],
        'kuramoto_alpha': [], 'kuramoto_R2': [], 'kuramoto_Kc': [],
        'haus_alpha': [], 'haus_R2': [],
        'cascade_alpha': [], 'cascade_R2': [],
        'cascade_ode_alpha': [], 'cascade_ode_R2': [],
        'analytical_kur_alpha': [], 'analytical_lin_alpha': [],
        'analytical_quad_alpha': [],
    }

    # ======================================================================
    # PART A: Analytical multi-exponential (fast, exact)
    # ======================================================================
    print("\n" + "=" * 78)
    print("PART A: ANALYTICAL MULTI-EXPONENTIAL EIGENVALUE SPECTRA")
    print("=" * 78)
    print(f"\n{'N':>4} | {'Predicted':>10} | {'Kuramoto':>10} | {'Linear':>10} | {'Quadratic':>10}")
    print("-" * 60)

    for N in N_values:
        pred = log_prediction(N)

        a_kur, r2_kur, _ = analytical_multiexp_alpha(N, 'kuramoto')
        a_lin, r2_lin, _ = analytical_multiexp_alpha(N, 'linear')
        a_quad, r2_quad, _ = analytical_multiexp_alpha(N, 'quadratic')

        results['N'].append(N)
        results['prediction'].append(pred)
        results['analytical_kur_alpha'].append(a_kur)
        results['analytical_lin_alpha'].append(a_lin)
        results['analytical_quad_alpha'].append(a_quad)

        print(f"{N:4d} | {pred:10.6f} | {a_kur:10.6f} | {a_lin:10.6f} | {a_quad:10.6f}")

    # ======================================================================
    # PART B: Cooperative cascade ODE (fast, mean-field)
    # ======================================================================
    print("\n" + "=" * 78)
    print("PART B: COOPERATIVE CASCADE ODE (mean-field)")
    print("=" * 78)

    # Test at different K_rel values near threshold (K_rel = 1 is threshold)
    K_rel_values = [0.90, 0.95, 0.98, 0.99, 0.995, 0.999]
    print(f"\nN=12, testing at various K/K_c ratios:")
    print(f"{'K/Kc':>8} | {'alpha':>10} | {'tau':>10} | {'R2':>8}")
    print("-" * 45)

    for K_rel in K_rel_values:
        t, r = cooperative_ode_decay(12, K_rel, T_max=100)
        tau, alpha, A0, R2 = fit_kww(t, r)
        print(f"{K_rel:8.3f} | {alpha:10.6f} | {tau:10.4f} | {R2:8.5f}")

    print(f"\nSweep N at K/Kc = 0.999 (just below threshold):")
    print(f"{'N':>4} | {'Predicted':>10} | {'Cascade alpha':>14} | {'R2':>8}")
    print("-" * 45)

    for i, N in enumerate(N_values):
        pred = log_prediction(N)
        t, r = cooperative_ode_decay(N, 0.999, T_max=200)
        tau, alpha, A0, R2 = fit_kww(t, r)
        results['cascade_ode_alpha'].append(alpha)
        results['cascade_ode_R2'].append(R2)
        print(f"{N:4d} | {pred:10.6f} | {alpha:14.6f} | {R2:8.5f}")

    # ======================================================================
    # PART C: Stochastic cooperative cascade (Monte Carlo)
    # ======================================================================
    print("\n" + "=" * 78)
    print("PART C: STOCHASTIC COOPERATIVE CASCADE (Monte Carlo, 500 trials)")
    print("=" * 78)
    print(f"{'N':>4} | {'Predicted':>10} | {'MC alpha':>10} | {'R2':>8}")
    print("-" * 45)

    for i, N in enumerate(N_values):
        pred = log_prediction(N)
        t, r_mean, r_std = cooperative_cascade_decay(N, 0.999, T_max=30, n_trials=500)
        tau, alpha, A0, R2 = fit_kww(t, r_mean)
        results['cascade_alpha'].append(alpha)
        results['cascade_R2'].append(R2)
        print(f"{N:4d} | {pred:10.6f} | {alpha:10.6f} | {R2:8.5f}")

    # ======================================================================
    # PART D: Kuramoto model (dynamical, full simulation)
    # ======================================================================
    print("\n" + "=" * 78)
    print("PART D: KURAMOTO MODEL (full dynamical simulation)")
    print("=" * 78)

    for i, N in enumerate(N_values):
        pred = log_prediction(N)
        omega = (np.arange(N) - (N - 1) / 2.0) * dw

        print(f"\n--- N = {N} (prediction: {pred:.6f}) ---")
        sys.stdout.flush()

        # Find threshold
        K_c = kuramoto_threshold(N, dw)
        results['kuramoto_Kc'].append(K_c)
        print(f"  K_c = {K_c:.4f}")

        # Test at K = 0.95*K_c, 0.98*K_c, 0.99*K_c
        best_alpha = np.nan
        best_R2 = 0
        for frac in [0.95, 0.98, 0.99, 0.995]:
            K_test = frac * K_c
            t, r_mean, r_std = run_kuramoto_decay(N, K_test, omega,
                                                   T_max=min(300, 50 * N),
                                                   n_trials=40)
            tau, alpha, A0, R2 = fit_kww(t, r_mean)
            print(f"  K/Kc={frac:.3f}: alpha={alpha:.6f}, tau={tau:.4f}, R2={R2:.5f}")
            if R2 > best_R2 and np.isfinite(alpha):
                best_alpha = alpha
                best_R2 = R2

        results['kuramoto_alpha'].append(best_alpha)
        results['kuramoto_R2'].append(best_R2)
        sys.stdout.flush()

    # ======================================================================
    # PART E: Haus four-wave-mixing model
    # ======================================================================
    print("\n" + "=" * 78)
    print("PART E: HAUS FOUR-WAVE-MIXING MODEL")
    print("=" * 78)

    # Base parameters
    g0 = 1.0   # peak gain
    l0 = 0.9   # loss (close to gain for threshold regime)
    dw_wg = 0.15  # mode spacing / gain bandwidth

    for i, N in enumerate(N_values):
        pred = log_prediction(N)
        params_base = (g0, l0, dw_wg, 0.0)

        print(f"\n--- N = {N} (prediction: {pred:.6f}) ---")
        sys.stdout.flush()

        # Find threshold
        gamma_c = find_haus_threshold(N, params_base, gamma_range=(0.001, 1.0))
        print(f"  gamma_c = {gamma_c:.6f}")

        # Test at gamma slightly below threshold
        best_alpha = np.nan
        best_R2 = 0
        for frac in [0.90, 0.95, 0.98]:
            gamma_test = frac * gamma_c
            t, r_mean, r_std = run_haus_decay(N, gamma_test, params_base,
                                               T_max=200, n_trials=20)
            tau, alpha, A0, R2 = fit_kww(t, r_mean)
            print(f"  gamma/gamma_c={frac:.2f}: alpha={alpha:.6f}, tau={tau:.4f}, R2={R2:.5f}")
            if R2 > best_R2 and np.isfinite(alpha):
                best_alpha = alpha
                best_R2 = R2

        results['haus_alpha'].append(best_alpha)
        results['haus_R2'].append(best_R2)
        sys.stdout.flush()

    # ======================================================================
    # SUMMARY TABLE
    # ======================================================================
    print("\n" + "=" * 78)
    print("SUMMARY: alpha AT MODE-LOCKING THRESHOLD vs log_{N-1}(N)")
    print("=" * 78)
    print(f"\n{'N':>4} | {'Predicted':>10} | {'Kuramoto':>10} | {'Haus FWM':>10} | "
          f"{'Cascade MC':>10} | {'Cascade ODE':>12} | {'Analytic':>10}")
    print("-" * 85)

    for i, N in enumerate(N_values):
        pred = results['prediction'][i]
        k_a = results['kuramoto_alpha'][i]
        h_a = results['haus_alpha'][i]
        c_a = results['cascade_alpha'][i]
        co_a = results['cascade_ode_alpha'][i]
        an_a = results['analytical_kur_alpha'][i]
        print(f"{N:4d} | {pred:10.6f} | {k_a:10.6f} | {h_a:10.6f} | "
              f"{c_a:10.6f} | {co_a:12.6f} | {an_a:10.6f}")

    # ======================================================================
    # DEVIATION ANALYSIS
    # ======================================================================
    print("\n" + "=" * 78)
    print("DEVIATION FROM PREDICTION: |alpha_measured - log_{N-1}(N)| / log_{N-1}(N)")
    print("=" * 78)

    model_names = ['Kuramoto', 'Haus FWM', 'Cascade MC', 'Cascade ODE', 'Analytic']
    model_keys = ['kuramoto_alpha', 'haus_alpha', 'cascade_alpha',
                   'cascade_ode_alpha', 'analytical_kur_alpha']

    print(f"\n{'N':>4} | ", end='')
    for name in model_names:
        print(f"{name:>12} | ", end='')
    print()
    print("-" * (8 + 15 * len(model_names)))

    for i, N in enumerate(N_values):
        pred = results['prediction'][i]
        print(f"{N:4d} | ", end='')
        for key in model_keys:
            alpha = results[key][i]
            if np.isfinite(alpha) and pred > 0:
                dev = abs(alpha - pred) / pred
                print(f"{dev:12.4%} | ", end='')
            else:
                print(f"{'N/A':>12} | ", end='')
        print()

    # ======================================================================
    # SPECIAL ANALYSIS FOR N=12
    # ======================================================================
    print("\n" + "=" * 78)
    print("SPECIAL ANALYSIS: N=12")
    print("=" * 78)

    idx_12 = N_values.index(12)
    pred_12 = log_prediction(12)
    print(f"\nPrediction:     log_11(12) = {pred_12:.6f}")
    print(f"Note: delta = log_11(12) - 1 = {pred_12 - 1:.6f}")
    print(f"      10*delta = {10*(pred_12-1):.6f} (should be near 4/3 - 1 = 0.333)")
    print(f"      4/3 = {4/3:.6f}")
    print(f"      10*delta + 1 = {10*(pred_12-1)+1:.6f}")

    print(f"\nResults across all models for N=12:")
    for name, key in zip(model_names, model_keys):
        alpha = results[key][idx_12]
        if np.isfinite(alpha):
            dev_pct = 100 * abs(alpha - pred_12) / pred_12
            print(f"  {name:15s}: alpha = {alpha:.6f} (deviation = {dev_pct:.2f}%)")
        else:
            print(f"  {name:15s}: N/A")

    # ======================================================================
    # SCALING ANALYSIS: Does alpha follow log_{N-1}(N)?
    # ======================================================================
    print("\n" + "=" * 78)
    print("SCALING ANALYSIS: Is alpha(N) = log_{N-1}(N) universal?")
    print("=" * 78)

    for name, key in zip(model_names, model_keys):
        alphas = np.array(results[key])
        preds = np.array(results['prediction'])
        valid = np.isfinite(alphas) & np.isfinite(preds)
        if np.sum(valid) >= 3:
            # Pearson correlation
            r_corr = np.corrcoef(preds[valid], alphas[valid])[0, 1]
            # Mean absolute deviation
            mad = np.mean(np.abs(alphas[valid] - preds[valid]))
            # Mean relative deviation
            mrd = np.mean(np.abs(alphas[valid] - preds[valid]) / preds[valid])
            print(f"\n  {name}:")
            print(f"    Pearson r with prediction: {r_corr:.4f}")
            print(f"    Mean absolute deviation:   {mad:.6f}")
            print(f"    Mean relative deviation:   {mrd:.4%}")
            # Linear fit alpha = a * log_{N-1}(N) + b
            from numpy.polynomial import polynomial as P
            coeffs = np.polyfit(preds[valid], alphas[valid], 1)
            print(f"    Linear fit: alpha = {coeffs[0]:.4f} * log_{{N-1}}(N) + {coeffs[1]:.4f}")
            print(f"    (Perfect match: slope=1.0, intercept=0.0)")

    # ======================================================================
    # VERDICT
    # ======================================================================
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)

    # Check if any model gives alpha close to log_{N-1}(N) consistently
    threshold = 0.05  # 5% relative deviation
    for name, key in zip(model_names, model_keys):
        alphas = np.array(results[key])
        preds = np.array(results['prediction'])
        valid = np.isfinite(alphas) & np.isfinite(preds)
        if np.sum(valid) >= 3:
            rel_devs = np.abs(alphas[valid] - preds[valid]) / preds[valid]
            n_match = np.sum(rel_devs < threshold)
            n_total = np.sum(valid)
            if n_match == n_total:
                status = "CONFIRMED (all N values within 5%)"
            elif n_match > n_total / 2:
                status = f"PARTIAL ({n_match}/{n_total} within 5%)"
            else:
                status = f"NOT CONFIRMED ({n_match}/{n_total} within 5%)"

            # Special check for N=12
            idx = N_values.index(12)
            if np.isfinite(alphas[idx]):
                dev_12 = abs(alphas[idx] - pred_12) / pred_12
                n12_status = f"alpha_12 = {alphas[idx]:.4f}, dev = {dev_12:.2%}"
            else:
                n12_status = "N/A"

            print(f"\n  {name}:")
            print(f"    Overall: {status}")
            print(f"    N=12:    {n12_status}")

    print("\n" + "=" * 78)
    print("KEY QUESTION ANSWERS:")
    print("=" * 78)
    print(f"""
1. Is the mode-locking threshold exponent log_11(12)?
   -> See results above for each model.

2. Is log_{{N-1}}(N) a universal scaling across all N?
   -> See scaling analysis above.

3. If confirmed: log_11(12) is a mathematical constant of standing wave
   formation, derivable from phase synchronisation theory alone.

4. If refuted: the threshold exponent depends on model details (coupling
   structure, amplitude dynamics, etc.) and is NOT a universal constant.
   In this case, log_11(12) may still be specific to the E6 geometry
   (12-step ouroboros) but is not derivable from generic mode-locking.
""")


if __name__ == '__main__':
    run_full_analysis()
