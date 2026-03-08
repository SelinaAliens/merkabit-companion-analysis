#!/usr/bin/env python3
"""
TESSERACT ZERO POINT CONSTANT INVESTIGATION
=============================================

Find the geometric measure of self-sustaining coherence in the tesseract.

Hypothesis: The tesseract (16 vertices, 4D self-dual hypercube) is the first
geometric structure where the standing wave is self-sustaining without external
drive. The ouroboros loop is internal to the geometry itself.

Tasks:
  2. Self-sustaining coherence test (drive removal)
  3. Internal ouroboros loop and Berry phase
  4. Zero point basin geometry
  5. 16-vertex vs 8-vertex comparison
  6. Candidate constant identification
  7. Octeract connection and dimensional ladder
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import FIGURES_DIR

RESULTS_DIR = Path(FIGURES_DIR)

np.random.seed(42)
COXETER_H = 12
STEP_PHASE = 2 * np.pi / COXETER_H
OUROBOROS_GATES = ['S', 'R', 'T', 'F', 'P']
NUM_GATES = 5

# ============================================================================
# 2-SPINOR MERKABIT (dual tetrahedra, 8 vertices)
# ============================================================================

class Merkabit2:
    """2-spinor merkabit on S3 x S3."""
    def __init__(self, u, v):
        self.u = np.array(u, dtype=complex)
        self.v = np.array(v, dtype=complex)
        self.u /= np.linalg.norm(self.u)
        self.v /= np.linalg.norm(self.v)

    @property
    def coherence(self):
        return np.real(np.vdot(self.u, self.v))

    @property
    def overlap_magnitude(self):
        return abs(np.vdot(self.u, self.v))

    def copy(self):
        return Merkabit2(self.u.copy(), self.v.copy())


def step_2_full(state, k):
    """Full ouroboros step for 2-spinor (with external drive)."""
    theta = STEP_PHASE
    absent = k % NUM_GATES
    p_angle = theta
    sym_base = theta / 3
    omega_k = 2 * np.pi * k / COXETER_H
    rx_a = sym_base * (1.0 + 0.5 * np.cos(omega_k))
    rz_a = sym_base * (1.0 + 0.5 * np.cos(omega_k + 2*np.pi/3))
    gl = OUROBOROS_GATES[absent]
    if gl == 'S': rz_a *= 0.4; rx_a *= 1.3
    elif gl == 'R': rx_a *= 0.4; rz_a *= 1.3
    elif gl == 'T': rx_a *= 0.7; rz_a *= 0.7
    elif gl == 'P': p_angle *= 0.6; rx_a *= 1.8; rz_a *= 1.5

    Pf = np.diag([np.exp(1j*p_angle/2), np.exp(-1j*p_angle/2)])
    Pi = np.diag([np.exp(-1j*p_angle/2), np.exp(1j*p_angle/2)])
    Rz = np.diag([np.exp(-1j*rz_a/2), np.exp(1j*rz_a/2)])
    c, ss = np.cos(rx_a/2), -1j * np.sin(rx_a/2)
    Rx = np.array([[c, ss], [ss, c]], dtype=complex)

    u = Rx @ Rz @ Pf @ state.u
    v = Rx @ Rz @ Pi @ state.v
    return Merkabit2(u, v)


# ============================================================================
# 4-SPINOR MERKABIT (tesseract, 16 vertices)
# ============================================================================

class Merkabit4:
    """4-spinor merkabit on S7 x S7."""
    def __init__(self, u, v):
        self.u = np.array(u, dtype=complex).flatten()
        self.v = np.array(v, dtype=complex).flatten()
        assert len(self.u) == 4 and len(self.v) == 4
        self.u /= np.linalg.norm(self.u)
        self.v /= np.linalg.norm(self.v)

    @property
    def coherence(self):
        return np.real(np.vdot(self.u, self.v))

    @property
    def overlap_magnitude(self):
        return abs(np.vdot(self.u, self.v))

    @property
    def overlap(self):
        return np.vdot(self.u, self.v)

    @property
    def relative_phase(self):
        return np.angle(self.overlap)

    @property
    def internal_entanglement(self):
        U_mat = self.u.reshape(2, 2)
        V_mat = self.v.reshape(2, 2)
        su = np.linalg.svd(U_mat, compute_uv=False)
        sv = np.linalg.svd(V_mat, compute_uv=False)
        su_n = su / np.sum(su)
        sv_n = sv / np.sum(sv)
        ent_u = -np.sum(su_n**2 * np.log2(su_n**2 + 1e-15))
        ent_v = -np.sum(sv_n**2 * np.log2(sv_n**2 + 1e-15))
        return (ent_u + ent_v) / 2

    def copy(self):
        return Merkabit4(self.u.copy(), self.v.copy())


# ============================================================================
# 8-SPINOR MERKABIT (octeract, 256 vertices)
# ============================================================================

class Merkabit8:
    """8-spinor merkabit on S15 x S15."""
    def __init__(self, u, v):
        self.u = np.array(u, dtype=complex).flatten()
        self.v = np.array(v, dtype=complex).flatten()
        assert len(self.u) == 8 and len(self.v) == 8
        self.u /= np.linalg.norm(self.u)
        self.v /= np.linalg.norm(self.v)

    @property
    def coherence(self):
        return np.real(np.vdot(self.u, self.v))

    @property
    def overlap_magnitude(self):
        return abs(np.vdot(self.u, self.v))

    def copy(self):
        return Merkabit8(self.u.copy(), self.v.copy())


# ============================================================================
# ROTATION GENERATORS
# ============================================================================

def rotation_matrix_4(plane, theta):
    """
    Generate a 4x4 rotation matrix in a given plane.
    plane: tuple (i, j) indicating the rotation plane (0-indexed)
    theta: rotation angle
    """
    R = np.eye(4, dtype=complex)
    c, s = np.cos(theta), np.sin(theta)
    R[plane[0], plane[0]] = c
    R[plane[0], plane[1]] = -s
    R[plane[1], plane[0]] = s
    R[plane[1], plane[1]] = c
    return R


def isoclinic_rotation_4(plane_pair, theta):
    """
    Isoclinic rotation: simultaneous rotation in two orthogonal planes.
    plane_pair: index 0,1,2 for the three pairs:
      0: (0,2)+(1,3)  — the existing cross gate
      1: (0,1)+(2,3)  — horizontal coupling
      2: (0,3)+(1,2)  — diagonal coupling
    """
    pairs = [
        ((0, 2), (1, 3)),  # cross: connects upper to lower
        ((0, 1), (2, 3)),  # horizontal: within-sector coupling
        ((0, 3), (1, 2)),  # diagonal: skew coupling
    ]
    p1, p2 = pairs[plane_pair]
    R1 = rotation_matrix_4(p1, theta)
    R2 = rotation_matrix_4(p2, theta)
    # Combined isoclinic rotation (compose both)
    return R2 @ R1


def cross_gate_4(theta, asymmetric=True):
    """Cross-coupling gate from original simulation."""
    c, s = np.cos(theta/2), np.sin(theta/2)
    Cf = np.array([
        [c, 0, -s, 0],
        [0, c, 0, -s],
        [s, 0, c, 0],
        [0, s, 0, c],
    ], dtype=complex)
    if asymmetric:
        Ci = np.array([
            [c, 0, s, 0],
            [0, c, 0, s],
            [-s, 0, c, 0],
            [0, -s, 0, c],
        ], dtype=complex)
    else:
        Ci = Cf.copy()
    return Cf, Ci


def cross_gate_horizontal_4(theta, asymmetric=True):
    """Horizontal coupling: (0,1)+(2,3) planes."""
    c, s = np.cos(theta/2), np.sin(theta/2)
    Cf = np.array([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, c, -s],
        [0, 0, s, c],
    ], dtype=complex)
    if asymmetric:
        Ci = np.array([
            [c, s, 0, 0],
            [-s, c, 0, 0],
            [0, 0, c, s],
            [0, 0, -s, c],
        ], dtype=complex)
    else:
        Ci = Cf.copy()
    return Cf, Ci


def cross_gate_diagonal_4(theta, asymmetric=True):
    """Diagonal coupling: (0,3)+(1,2) planes."""
    c, s = np.cos(theta/2), np.sin(theta/2)
    Cf = np.array([
        [c, 0, 0, -s],
        [0, c, -s, 0],
        [0, s, c, 0],
        [s, 0, 0, c],
    ], dtype=complex)
    if asymmetric:
        Ci = np.array([
            [c, 0, 0, s],
            [0, c, s, 0],
            [0, -s, c, 0],
            [-s, 0, 0, c],
        ], dtype=complex)
    else:
        Ci = Cf.copy()
    return Cf, Ci


# ============================================================================
# INTERNAL DYNAMICS (no external drive)
# ============================================================================

def internal_step_4(state, step_index, coupling_strength=1.0):
    """
    Evolve under INTERNAL tesseract geometry only.
    No P, Rx, Rz gates (external drive removed).
    Only cross-coupling between sectors (the 4D structure).

    Cycles through 3 isoclinic rotation planes with triality phase.
    """
    theta = STEP_PHASE * coupling_strength
    omega_k = 2 * np.pi * step_index / COXETER_H

    # Cycle through 3 coupling planes with triality modulation
    # This gives the tesseract its full rotational freedom
    theta_cross = theta * (1.0 + 0.3 * np.cos(omega_k))
    theta_horiz = theta * (1.0 + 0.3 * np.cos(omega_k + 2*np.pi/3))
    theta_diag = theta * (1.0 + 0.3 * np.cos(omega_k + 4*np.pi/3))

    u, v = state.u.copy(), state.v.copy()

    # Cross coupling (0,2)+(1,3) — asymmetric for counter-rotation
    Cf, Ci = cross_gate_4(theta_cross)
    u = Cf @ u
    v = Ci @ v

    # Horizontal coupling (0,1)+(2,3)
    Hf, Hi = cross_gate_horizontal_4(theta_horiz)
    u = Hf @ u
    v = Hi @ v

    # Diagonal coupling (0,3)+(1,2)
    Df, Di = cross_gate_diagonal_4(theta_diag)
    u = Df @ u
    v = Di @ v

    return Merkabit4(u, v)


def internal_step_8(state, step_index, coupling_strength=1.0):
    """
    Internal dynamics for 8-spinor (octeract).
    Cross-coupling between all pairs of 2-spinor sectors.
    7 independent cross-coupling planes in 8D.
    """
    theta = STEP_PHASE * coupling_strength
    omega_k = 2 * np.pi * step_index / COXETER_H

    u, v = state.u.copy(), state.v.copy()

    # 8D has C(8,2)/2 = 28/2 = 7 independent isoclinic rotation planes
    # (organized as pairs of orthogonal 2-planes)
    # Use the 7 natural pairs from the octonion multiplication table
    pairs = [
        ((0, 4), (1, 5)),  # L/R coupling, sector 0+1
        ((2, 6), (3, 7)),  # L/R coupling, sector 2+3
        ((0, 2), (1, 3)),  # Within-L cross
        ((4, 6), (5, 7)),  # Within-R cross
        ((0, 1), (2, 3)),  # Within-sector-0 horizontal
        ((4, 5), (6, 7)),  # Within-sector-2 horizontal
        ((0, 7), (3, 4)),  # Long diagonal (octonionic)
    ]

    for idx, (p1, p2) in enumerate(pairs):
        phase = omega_k + 2 * np.pi * idx / 7
        th = theta * (1.0 + 0.3 * np.cos(phase))

        # Build 8x8 rotation in planes p1 and p2
        c, s = np.cos(th/2), np.sin(th/2)

        Rf = np.eye(8, dtype=complex)
        Rf[p1[0], p1[0]] = c; Rf[p1[0], p1[1]] = -s
        Rf[p1[1], p1[0]] = s; Rf[p1[1], p1[1]] = c
        Rf[p2[0], p2[0]] = c; Rf[p2[0], p2[1]] = -s
        Rf[p2[1], p2[0]] = s; Rf[p2[1], p2[1]] = c

        Ri = np.eye(8, dtype=complex)
        Ri[p1[0], p1[0]] = c; Ri[p1[0], p1[1]] = s
        Ri[p1[1], p1[0]] = -s; Ri[p1[1], p1[1]] = c
        Ri[p2[0], p2[0]] = c; Ri[p2[0], p2[1]] = s
        Ri[p2[1], p2[0]] = -s; Ri[p2[1], p2[1]] = c

        u = Rf @ u
        v = Ri @ v

    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    return Merkabit8(u, v)


# ============================================================================
# FULL OUROBOROS STEP (4-spinor, for comparison)
# ============================================================================

def ouroboros_step_4_full(state, step_index, cross_strength=0.3):
    """Full ouroboros step with external drive (from original simulation)."""
    theta = STEP_PHASE
    k = step_index
    absent = k % NUM_GATES
    p_angle = theta
    sym_base = theta / 3
    omega_k = 2 * np.pi * k / COXETER_H
    rx_angle = sym_base * (1.0 + 0.5 * np.cos(omega_k))
    rz_angle = sym_base * (1.0 + 0.5 * np.cos(omega_k + 2*np.pi/3))
    cross_angle = cross_strength * theta * (1.0 + 0.5 * np.cos(omega_k + 4*np.pi/3))

    gl = OUROBOROS_GATES[absent]
    if gl == 'S': rz_angle *= 0.4; rx_angle *= 1.3; cross_angle *= 1.2
    elif gl == 'R': rx_angle *= 0.4; rz_angle *= 1.3; cross_angle *= 0.8
    elif gl == 'T': rx_angle *= 0.7; rz_angle *= 0.7; cross_angle *= 1.5
    elif gl == 'P': p_angle *= 0.6; rx_angle *= 1.8; rz_angle *= 1.5; cross_angle *= 0.5

    u, v = state.u.copy(), state.v.copy()

    # P gate (asymmetric phase)
    P2f = np.diag([np.exp(1j*p_angle/2), np.exp(-1j*p_angle/2)])
    P2i = np.diag([np.exp(-1j*p_angle/2), np.exp(1j*p_angle/2)])
    Pf = np.block([[P2f, np.zeros((2,2))], [np.zeros((2,2)), P2f]])
    Pi = np.block([[P2i, np.zeros((2,2))], [np.zeros((2,2)), P2i]])
    u = Pf @ u; v = Pi @ v

    # Cross gate (asymmetric)
    Cf, Ci = cross_gate_4(cross_angle)
    u = Cf @ u; v = Ci @ v

    # Rz
    R2z = np.diag([np.exp(-1j*rz_angle/2), np.exp(1j*rz_angle/2)])
    R4z = np.block([[R2z, np.zeros((2,2))], [np.zeros((2,2)), R2z]])
    u = R4z @ u; v = R4z @ v

    # Rx
    c, ss = np.cos(rx_angle/2), -1j * np.sin(rx_angle/2)
    R2x = np.array([[c, ss], [ss, c]], dtype=complex)
    R4x = np.block([[R2x, np.zeros((2,2))], [np.zeros((2,2)), R2x]])
    u = R4x @ u; v = R4x @ v

    return Merkabit4(u, v)


# ============================================================================
# BERRY PHASE COMPUTATION
# ============================================================================

def berry_phase(u_list, v_list):
    """Compute Berry phase from a sequence of (u,v) states."""
    n = len(u_list)
    gamma = 0.0
    for k in range(n):
        k_next = (k + 1) % n
        ou = np.vdot(u_list[k], u_list[k_next])
        ov = np.vdot(v_list[k], v_list[k_next])
        gamma += np.angle(ou * ov)
    return -gamma


# ============================================================================
# TASK 2: SELF-SUSTAINING COHERENCE TEST
# ============================================================================

def task2_self_sustaining_coherence():
    print("\n" + "=" * 76)
    print("TASK 2: SELF-SUSTAINING COHERENCE TEST")
    print("  Initialize pi-lock standing wave, remove external drive,")
    print("  evolve under internal geometry alone")
    print("=" * 76)

    n_steps = 500  # number of evolution steps

    # --- 2-spinor (dual tetrahedra): NO internal geometry ---
    # Standing wave: u = [1,0], v = [0,1] (orthogonal)
    s2 = Merkabit2([1, 0], [0, 1])
    c2_initial = s2.coherence
    # Without drive, 2-spinor has NO internal dynamics -> state frozen
    c2_t = np.full(n_steps + 1, c2_initial)

    # --- 4-spinor (tesseract): internal cross-coupling ---
    # Standing wave variants
    standing_waves_4 = {
        '|0> simple': Merkabit4([1, 0, 0, 0], [0, 0, 0, 1]),
        '|0> spread': Merkabit4([1/np.sqrt(2), 1/np.sqrt(2), 0, 0],
                                [0, 0, 1/np.sqrt(2), 1/np.sqrt(2)]),
        '|0> tesseract': Merkabit4([1, 1, 1, 1], [1, -1, -1, 1]),
        '|0> pi-lock': Merkabit4([1, 0, 0, 0], [-1, 0, 0, 0]),  # phi = pi
    }

    results_4 = {}
    for label, s0 in standing_waves_4.items():
        s = s0.copy()
        ct = [s.coherence]
        ent_t = [s.internal_entanglement]
        overlap_t = [s.overlap_magnitude]
        for step in range(n_steps):
            s = internal_step_4(s, step % COXETER_H)
            ct.append(s.coherence)
            ent_t.append(s.internal_entanglement)
            overlap_t.append(s.overlap_magnitude)
        results_4[label] = {
            'C': np.array(ct),
            'ent': np.array(ent_t),
            'overlap': np.array(overlap_t),
        }

    # --- 8-spinor (octeract): deeper internal geometry ---
    s8_0 = Merkabit8([1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1])
    s = s8_0.copy()
    c8_t = [s.coherence]
    for step in range(n_steps):
        s = internal_step_8(s, step % COXETER_H)
        c8_t.append(s.coherence)
    c8_t = np.array(c8_t)

    # Print results
    print(f"\n  Evolution over {n_steps} steps (no external drive)")
    print(f"\n  {'System':<20}  {'C(0)':<8}  {'C(100)':<8}  {'C(250)':<8}  "
          f"{'C(500)':<8}  {'|C|_mean':<10}  {'Status':<20}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*8}  "
          f"{'-'*8}  {'-'*10}  {'-'*20}")

    # 2-spinor
    print(f"  {'2-spinor (frozen)':<20}  {c2_t[0]:<8.4f}  {c2_t[100]:<8.4f}  "
          f"{c2_t[250]:<8.4f}  {c2_t[500]:<8.4f}  "
          f"{np.mean(np.abs(c2_t)):<10.4f}  {'FROZEN (no dynamics)':<20}")

    for label, res in results_4.items():
        ct = res['C']
        mean_abs_c = np.mean(np.abs(ct))
        status = "SUSTAINED" if mean_abs_c > 0.3 else ("OSCILLATING" if np.std(ct) > 0.1 else "DECAYED")
        print(f"  {label:<20}  {ct[0]:<8.4f}  {ct[100]:<8.4f}  "
              f"{ct[250]:<8.4f}  {ct[500]:<8.4f}  "
              f"{mean_abs_c:<10.4f}  {status:<20}")

    # 8-spinor
    mean_abs_c8 = np.mean(np.abs(c8_t))
    status8 = "SUSTAINED" if mean_abs_c8 > 0.3 else ("OSCILLATING" if np.std(c8_t) > 0.1 else "DECAYED")
    print(f"  {'8-spinor':<20}  {c8_t[0]:<8.4f}  {c8_t[100]:<8.4f}  "
          f"{c8_t[250]:<8.4f}  {c8_t[500]:<8.4f}  "
          f"{mean_abs_c8:<10.4f}  {status8:<20}")

    # Plot C(t) for all systems
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    t = np.arange(n_steps + 1)

    ax = axes[0]
    ax.plot(t, c2_t, 'k--', label='2-spinor (frozen)', alpha=0.5)
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    for (label, res), col in zip(results_4.items(), colors):
        ax.plot(t, res['C'], color=col, label=f'4-spinor {label}', alpha=0.8)
    ax.plot(t, c8_t, 'orange', label='8-spinor', alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('C(t) = Re(u^dagger v)')
    ax.set_title('TASK 2: Coherence Under Internal Dynamics Only (No External Drive)')
    ax.legend(fontsize=8)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for (label, res), col in zip(results_4.items(), colors):
        ax.plot(t, res['overlap'], color=col, label=f'|u^dv| {label}', alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('|u^dagger v| (overlap magnitude)')
    ax.set_title('Overlap Magnitude Under Internal Dynamics')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'task2_coherence_internal_dynamics.png', dpi=150)
    plt.close()
    print(f"\n  Plot saved: task2_coherence_internal_dynamics.png")

    return results_4, c2_t, c8_t


# ============================================================================
# TASK 3: INTERNAL OUROBOROS LOOP AND BERRY PHASE
# ============================================================================

def task3_internal_berry_phase():
    print("\n" + "=" * 76)
    print("TASK 3: INTERNAL OUROBOROS LOOP")
    print("  Find the self-referential loop inside the tesseract geometry")
    print("  Compute Berry phase of the internal closed path")
    print("=" * 76)

    # The tesseract has 3 pairs of orthogonal rotation planes in 4D.
    # An internal ouroboros loop cycles through all 3, returning to start.
    #
    # Three isoclinic generators:
    #   L1: (0,2)+(1,3)  L2: (0,1)+(2,3)  L3: (0,3)+(1,2)
    #
    # A triangular loop in parameter space:
    #   Apply L1(theta) -> L2(theta) -> L3(theta)
    #   The product is NOT identity for generic theta.
    #   The Berry phase = total accumulated geometric phase.

    print(f"\n  --- Berry phase of internal triangular loops ---")
    print(f"  Loop: L1(theta) -> L2(theta) -> L3(theta)")

    # Test different loop sizes (theta values)
    thetas = np.linspace(0.01, 2*np.pi, 200)
    berry_phases_simple = []
    berry_phases_tesseract = []
    berry_phases_spread = []

    # Initial states
    u0_simple = np.array([1, 0, 0, 0], dtype=complex)
    v0_simple = np.array([0, 0, 0, 1], dtype=complex)

    u0_tess = np.array([1, 1, 1, 1], dtype=complex) / 2
    v0_tess = np.array([1, -1, -1, 1], dtype=complex) / 2

    u0_spread = np.array([1, 1, 0, 0], dtype=complex) / np.sqrt(2)
    v0_spread = np.array([0, 0, 1, 1], dtype=complex) / np.sqrt(2)

    for theta in thetas:
        for u0, v0, phases_list in [(u0_simple, v0_simple, berry_phases_simple),
                                      (u0_tess, v0_tess, berry_phases_tesseract),
                                      (u0_spread, v0_spread, berry_phases_spread)]:
            u_states, v_states = [u0.copy()], [v0.copy()]
            u, v = u0.copy(), v0.copy()

            # Step 1: L1 (cross: (0,2)+(1,3))
            n_sub = 12
            for i in range(n_sub):
                th = theta * (i + 1) / n_sub
                Cf, Ci = cross_gate_4(th / n_sub)
                u = Cf @ u; v = Ci @ v
                u /= np.linalg.norm(u); v /= np.linalg.norm(v)
                u_states.append(u.copy()); v_states.append(v.copy())

            # Step 2: L2 (horizontal: (0,1)+(2,3))
            for i in range(n_sub):
                th = theta * (i + 1) / n_sub
                Hf, Hi = cross_gate_horizontal_4(th / n_sub)
                u = Hf @ u; v = Hi @ v
                u /= np.linalg.norm(u); v /= np.linalg.norm(v)
                u_states.append(u.copy()); v_states.append(v.copy())

            # Step 3: L3 (diagonal: (0,3)+(1,2))
            for i in range(n_sub):
                th = theta * (i + 1) / n_sub
                Df, Di = cross_gate_diagonal_4(th / n_sub)
                u = Df @ u; v = Di @ v
                u /= np.linalg.norm(u); v /= np.linalg.norm(v)
                u_states.append(u.copy()); v_states.append(v.copy())

            gamma = berry_phase(u_states, v_states)
            phases_list.append(gamma)

    berry_phases_simple = np.array(berry_phases_simple)
    berry_phases_tesseract = np.array(berry_phases_tesseract)
    berry_phases_spread = np.array(berry_phases_spread)

    # Find the theta where loop closes (state returns closest to initial)
    print(f"\n  {'theta/pi':<10}  {'gamma_simple':<14}  {'gamma/2pi':<12}  "
          f"{'gamma_tess':<14}  {'gamma/2pi':<12}  "
          f"{'gamma_spread':<14}  {'gamma/2pi':<12}")
    print(f"  {'-'*10}  {'-'*14}  {'-'*12}  "
          f"{'-'*14}  {'-'*12}  "
          f"{'-'*14}  {'-'*12}")

    # Print at selected theta values
    for idx in [0, 24, 49, 74, 99, 124, 149, 174, 199]:
        th = thetas[idx]
        gs = berry_phases_simple[idx]
        gt = berry_phases_tesseract[idx]
        gsp = berry_phases_spread[idx]
        print(f"  {th/np.pi:<10.4f}  {gs:<14.6f}  {gs/(2*np.pi):<12.6f}  "
              f"{gt:<14.6f}  {gt/(2*np.pi):<12.6f}  "
              f"{gsp:<14.6f}  {gsp/(2*np.pi):<12.6f}")

    # --- FULL OUROBOROS INTERNAL CYCLE ---
    # Run one complete Coxeter cycle (12 steps) using only internal dynamics
    print(f"\n  --- Full Coxeter internal cycle (12 steps, internal only) ---")

    for label, u0, v0 in [('|0> simple', u0_simple, v0_simple),
                           ('|0> tesseract', u0_tess, v0_tess),
                           ('|0> spread', u0_spread, v0_spread)]:
        s = Merkabit4(u0, v0)
        u_states = [s.u.copy()]
        v_states = [s.v.copy()]
        for step in range(COXETER_H):
            s = internal_step_4(s, step)
            u_states.append(s.u.copy())
            v_states.append(s.v.copy())

        gamma = berry_phase(u_states[:-1], v_states[:-1])
        # Closure
        diff_u = np.linalg.norm(u_states[-1] - u_states[0])
        diff_v = np.linalg.norm(v_states[-1] - v_states[0])
        print(f"  {label:<15}: gamma = {gamma:.6f} rad = {gamma/np.pi:.6f}pi"
              f"  gamma/(2pi) = {gamma/(2*np.pi):.6f}"
              f"  closure: |du|={diff_u:.4f}, |dv|={diff_v:.4f}")

    # --- MULTI-CYCLE ACCUMULATION ---
    print(f"\n  --- Multi-cycle Berry phase accumulation (internal only) ---")
    for label, u0, v0 in [('|0> simple', u0_simple, v0_simple),
                           ('|0> tesseract', u0_tess, v0_tess)]:
        gammas_cumulative = []
        s = Merkabit4(u0, v0)
        u_all = [s.u.copy()]
        v_all = [s.v.copy()]
        for cycle in range(20):
            for step in range(COXETER_H):
                s = internal_step_4(s, step)
                u_all.append(s.u.copy())
                v_all.append(s.v.copy())
            gamma_c = berry_phase(u_all, v_all)
            gammas_cumulative.append(gamma_c)

        print(f"  {label}:")
        for cyc in [0, 1, 4, 9, 19]:
            g = gammas_cumulative[cyc]
            print(f"    Cycle {cyc+1:>3}: gamma = {g:>10.6f} rad  "
                  f"gamma/(2pi) = {g/(2*np.pi):>10.6f}  "
                  f"gamma/pi = {g/np.pi:>10.6f}")

    # --- Test specific values ---
    print(f"\n  --- Testing gamma/(2pi) against known constants ---")
    # Use the Coxeter cycle Berry phase for |0> simple
    s = Merkabit4(u0_simple, v0_simple)
    u_states = [s.u.copy()]
    v_states = [s.v.copy()]
    for step in range(COXETER_H):
        s = internal_step_4(s, step)
        u_states.append(s.u.copy())
        v_states.append(s.v.copy())

    gamma_ref = berry_phase(u_states[:-1], v_states[:-1])
    frac_wind = gamma_ref / (2 * np.pi)

    candidates = {
        '1/137': 1/137.036,
        '4/3': 4/3,
        '1/2': 0.5,
        '1/4': 0.25,
        'pi/12': np.pi/12,
        '1/12': 1/12,
        '1/3': 1/3,
        '2/3': 2/3,
        '1/6': 1/6,
        'pi/4': np.pi/4,
        'pi/6': np.pi/6,
        '1/(4pi)': 1/(4*np.pi),
        'ln2': np.log(2),
        '1/sqrt(2)': 1/np.sqrt(2),
        'sqrt(3)/2': np.sqrt(3)/2,
    }

    print(f"  Internal Berry phase: gamma = {gamma_ref:.6f} rad")
    print(f"  Fractional winding: gamma/(2pi) = {frac_wind:.6f}")
    print(f"  gamma/pi = {gamma_ref/np.pi:.6f}")
    print()
    print(f"  {'Constant':<15}  {'Value':<12}  {'|diff from gamma/(2pi)|':<25}  {'Match?'}")
    print(f"  {'-'*15}  {'-'*12}  {'-'*25}  {'-'*8}")
    for name, val in candidates.items():
        diff = abs(frac_wind - val)
        match = "YES" if diff < 0.01 else ("CLOSE" if diff < 0.05 else "no")
        print(f"  {name:<15}  {val:<12.6f}  {diff:<25.6f}  {match}")

    # Plot Berry phase vs theta
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(thetas/np.pi, berry_phases_simple/(2*np.pi), 'b-', label='|0> simple', linewidth=1.5)
    ax.plot(thetas/np.pi, berry_phases_tesseract/(2*np.pi), 'r-', label='|0> tesseract', linewidth=1.5)
    ax.plot(thetas/np.pi, berry_phases_spread/(2*np.pi), 'g-', label='|0> spread', linewidth=1.5)
    # Reference lines
    for name, val in [('1/4', 0.25), ('1/2', 0.5), ('1/3', 1/3), ('1/6', 1/6)]:
        ax.axhline(y=val, color='gray', linestyle='--', alpha=0.5, label=name)
        ax.axhline(y=-val, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('theta / pi')
    ax.set_ylabel('gamma / (2pi)')
    ax.set_title('TASK 3: Internal Berry Phase vs Loop Size')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'task3_internal_berry_phase.png', dpi=150)
    plt.close()
    print(f"\n  Plot saved: task3_internal_berry_phase.png")

    return gamma_ref, frac_wind


# ============================================================================
# TASK 4: ZERO POINT BASIN GEOMETRY
# ============================================================================

def task4_basin_geometry():
    print("\n" + "=" * 76)
    print("TASK 4: ZERO POINT BASIN GEOMETRY")
    print("  Volume fraction of state space where coherence is maintained")
    print("=" * 76)

    n_samples = 5000
    n_evolve = 100  # steps of internal evolution
    thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]

    def random_unit_vec(dim):
        v = np.random.randn(dim) + 1j * np.random.randn(dim)
        return v / np.linalg.norm(v)

    # --- 2-spinor basin ---
    # Without internal dynamics, coherence doesn't change -> basin = initial fraction
    c2_initial = []
    for _ in range(n_samples):
        u = random_unit_vec(2)
        v = random_unit_vec(2)
        c2_initial.append(abs(np.real(np.vdot(u, v))))

    # --- 4-spinor basin ---
    c4_initial = []
    c4_evolved = []
    c4_sustained = {th: 0 for th in thresholds}

    for i in range(n_samples):
        u = random_unit_vec(4)
        v = random_unit_vec(4)
        s = Merkabit4(u, v)
        c_init = abs(s.coherence)
        c4_initial.append(c_init)

        # Evolve under internal dynamics
        for step in range(n_evolve):
            s = internal_step_4(s, step % COXETER_H)
        c_final = abs(s.coherence)
        c4_evolved.append(c_final)

        for th in thresholds:
            if c_init < th and c_final < th:
                # Started near zero, stayed near zero -> in basin
                c4_sustained[th] += 1

    # --- 4-spinor: start from near-zero states ---
    c4_near_zero_sustained = {th: 0 for th in thresholds}
    n_near_zero = 5000
    n_near_zero_counted = {th: 0 for th in thresholds}

    for i in range(n_near_zero):
        # Generate state near |0> (low coherence)
        u = random_unit_vec(4)
        # Make v nearly orthogonal to u
        v = random_unit_vec(4)
        v = v - np.vdot(u, v) * u  # orthogonalize
        noise = 0.1 * random_unit_vec(4)  # small perturbation
        v = v + noise
        v /= np.linalg.norm(v)

        s = Merkabit4(u, v)
        c_init = abs(s.coherence)

        # Evolve
        coherences = [c_init]
        for step in range(n_evolve):
            s = internal_step_4(s, step % COXETER_H)
            coherences.append(abs(s.coherence))

        mean_c = np.mean(coherences)
        max_c = np.max(coherences)

        for th in thresholds:
            if c_init < th:
                n_near_zero_counted[th] += 1
                if max_c < th * 3:  # coherence doesn't explode
                    c4_near_zero_sustained[th] += 1

    # --- 8-spinor basin ---
    c8_near_zero_sustained = {th: 0 for th in thresholds}
    n8_counted = {th: 0 for th in thresholds}

    for i in range(min(n_near_zero, 2000)):  # fewer samples for speed
        u = random_unit_vec(8)
        v = random_unit_vec(8)
        v = v - np.vdot(u, v) * u
        noise = 0.1 * random_unit_vec(8)
        v = v + noise
        v /= np.linalg.norm(v)

        s = Merkabit8(u, v)
        c_init = abs(s.coherence)

        coherences = [c_init]
        for step in range(n_evolve):
            s = internal_step_8(s, step % COXETER_H)
            coherences.append(abs(s.coherence))

        max_c = np.max(coherences)

        for th in thresholds:
            if c_init < th:
                n8_counted[th] += 1
                if max_c < th * 3:
                    c8_near_zero_sustained[th] += 1

    # Print results
    print(f"\n  Random state sampling (n={n_samples})")
    print(f"\n  Fraction with |C| < threshold (random states on S^n x S^n):")
    print(f"  {'Threshold':<12}  {'2-spinor':<12}  {'4-spinor init':<15}  "
          f"{'4-spinor evolved':<18}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*15}  {'-'*18}")
    for th in thresholds:
        frac2 = np.sum(np.array(c2_initial) < th) / n_samples
        frac4i = np.sum(np.array(c4_initial) < th) / n_samples
        frac4e = np.sum(np.array(c4_evolved) < th) / n_samples
        print(f"  {th:<12.2f}  {frac2:<12.4f}  {frac4i:<15.4f}  {frac4e:<18.4f}")

    print(f"\n  Near-zero basin stability (start near |0>, evolve {n_evolve} steps):")
    print(f"  {'Threshold':<12}  {'4-spinor sustained':<22}  {'4-spinor fraction':<20}  "
          f"{'8-spinor sustained':<22}  {'8-spinor fraction':<20}")
    print(f"  {'-'*12}  {'-'*22}  {'-'*20}  {'-'*22}  {'-'*20}")

    basin_fractions_4 = {}
    basin_fractions_8 = {}
    for th in thresholds:
        n4 = n_near_zero_counted[th]
        sus4 = c4_near_zero_sustained[th]
        frac4 = sus4 / n4 if n4 > 0 else 0
        n8 = n8_counted[th]
        sus8 = c8_near_zero_sustained[th]
        frac8 = sus8 / n8 if n8 > 0 else 0
        basin_fractions_4[th] = frac4
        basin_fractions_8[th] = frac8
        print(f"  {th:<12.2f}  {sus4:>5}/{n4:<5}             "
              f"{frac4:<20.4f}  {sus8:>5}/{n8:<5}             "
              f"{frac8:<20.4f}")

    # Basin volume ratio: 4-spinor / 2-spinor
    print(f"\n  Basin volume ratios (4-spinor / 2-spinor equivalent):")
    for th in [0.1, 0.2, 0.3]:
        frac2 = np.sum(np.array(c2_initial) < th) / n_samples
        frac4 = basin_fractions_4.get(th, 0)
        ratio = frac4 / frac2 if frac2 > 0 else float('inf')
        print(f"    threshold={th}: ratio = {ratio:.4f}")

    return basin_fractions_4, basin_fractions_8


# ============================================================================
# TASK 5: 16-VERTEX vs 8-VERTEX COMPARISON
# ============================================================================

def task5_comparison():
    print("\n" + "=" * 76)
    print("TASK 5: 16-VERTEX vs 8-VERTEX COMPARISON")
    print("  Identical coherence decay experiments, drive removed")
    print("=" * 76)

    n_steps = 600

    # Initialize standing waves
    # 2-spinor: u=[1,0], v=[0,1]
    s2_init = Merkabit2([1, 0], [0, 1])

    # 4-spinor: u=[1,0,0,0], v=[0,0,0,1]
    s4_init = Merkabit4([1, 0, 0, 0], [0, 0, 0, 1])

    # Also test with partially coherent initial state (near standing wave)
    eps = 0.1
    s2_near = Merkabit2([1, 0], [eps, np.sqrt(1-eps**2)])
    s4_near = Merkabit4([1, 0, 0, 0], [eps, 0, 0, np.sqrt(1-eps**2)])

    # 8-spinor comparison
    s8_init = Merkabit8([1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1])

    # Evolve all under internal dynamics only
    # 2-spinor: frozen (no internal dynamics)
    c2_perfect = np.full(n_steps + 1, s2_init.coherence)
    c2_near = np.full(n_steps + 1, s2_near.coherence)

    # 4-spinor: internal cross-coupling
    s = s4_init.copy()
    c4_perfect = [s.coherence]
    for step in range(n_steps):
        s = internal_step_4(s, step % COXETER_H)
        c4_perfect.append(s.coherence)
    c4_perfect = np.array(c4_perfect)

    s = s4_near.copy()
    c4_near = [s.coherence]
    for step in range(n_steps):
        s = internal_step_4(s, step % COXETER_H)
        c4_near.append(s.coherence)
    c4_near = np.array(c4_near)

    # 8-spinor
    s = s8_init.copy()
    c8 = [s.coherence]
    for step in range(n_steps):
        s = internal_step_8(s, step % COXETER_H)
        c8.append(s.coherence)
    c8 = np.array(c8)

    # Also: DRIVEN comparison (with full ouroboros step)
    s = s4_init.copy()
    c4_driven = [s.coherence]
    for step in range(n_steps):
        s = ouroboros_step_4_full(s, step % COXETER_H)
        c4_driven.append(s.coherence)
    c4_driven = np.array(c4_driven)

    # Compute coherence lifetimes
    # Lifetime = number of steps until |C(t)| stays above some fraction of initial
    def coherence_lifetime(ct, threshold_frac=0.5):
        c0 = max(abs(ct[0]), 1e-10)
        threshold = c0 * threshold_frac
        # Find first time |C| drops below threshold and stays there
        for i in range(len(ct)):
            if abs(ct[i]) < threshold:
                # Check if it stays below
                window = ct[i:min(i+50, len(ct))]
                if np.mean(np.abs(window)) < threshold:
                    return i
        return len(ct)  # never drops

    # Print coherence evolution stats
    print(f"\n  Coherence C(t) evolution over {n_steps} steps:")
    print(f"\n  {'System':<25}  {'C(0)':<8}  {'std(C)':<10}  {'|C|_mean':<10}  "
          f"{'min(C)':<10}  {'max(C)':<10}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    for label, ct in [('2-spinor (frozen)', c2_perfect),
                       ('2-spinor near-zero', c2_near),
                       ('4-spinor internal', c4_perfect),
                       ('4-spinor near-zero int', c4_near),
                       ('4-spinor DRIVEN', c4_driven),
                       ('8-spinor internal', c8)]:
        print(f"  {label:<25}  {ct[0]:<8.4f}  {np.std(ct):<10.4f}  "
              f"{np.mean(np.abs(ct)):<10.4f}  {np.min(ct):<10.4f}  {np.max(ct):<10.4f}")

    # Coherence oscillation frequency analysis
    print(f"\n  Oscillation analysis (FFT of C(t)):")
    for label, ct in [('4-spinor internal', c4_perfect),
                       ('4-spinor near-zero', c4_near),
                       ('8-spinor internal', c8)]:
        if np.std(ct) > 0.001:
            fft = np.fft.rfft(ct - np.mean(ct))
            freqs = np.fft.rfftfreq(len(ct))
            power = np.abs(fft)**2
            peak_idx = np.argmax(power[1:]) + 1  # skip DC
            peak_freq = freqs[peak_idx]
            peak_period = 1/peak_freq if peak_freq > 0 else float('inf')
            print(f"  {label:<25}: peak freq = {peak_freq:.4f}, "
                  f"period = {peak_period:.1f} steps, "
                  f"period/12 = {peak_period/12:.2f} Coxeter cycles")

    # Lifetime ratio
    print(f"\n  Coherence lifetime comparison:")
    lt_4 = coherence_lifetime(c4_near, 0.5)
    lt_8 = coherence_lifetime(c8, 0.5)
    print(f"  4-spinor near-zero: {lt_4} steps ('{'>'+str(n_steps) if lt_4 >= n_steps else lt_4}')")
    print(f"  8-spinor: {lt_8} steps ('{'>'+str(n_steps) if lt_8 >= n_steps else lt_8}')")
    if lt_4 > 0 and lt_4 < n_steps and lt_8 < n_steps:
        print(f"  Ratio (8/4): {lt_8/lt_4:.4f}")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    t = np.arange(n_steps + 1)

    ax = axes[0]
    ax.plot(t, c2_perfect, 'k--', label='2-spinor (frozen)', alpha=0.5)
    ax.plot(t, c4_perfect, 'b-', label='4-spinor internal', alpha=0.8)
    ax.plot(t, c8, 'r-', label='8-spinor internal', alpha=0.8)
    ax.set_ylabel('C(t)')
    ax.set_title('TASK 5: Standing Wave Coherence - No External Drive')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t, c4_near, 'b-', label='4-spinor near-zero internal', alpha=0.8)
    ax.plot(t, c2_near, 'k--', label='2-spinor near-zero (frozen)', alpha=0.5)
    ax.plot(t, c4_driven, 'r-', label='4-spinor DRIVEN (full ouroboros)', alpha=0.6)
    ax.set_ylabel('C(t)')
    ax.set_title('Near-Zero Initial State: Internal vs Driven')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    # Power spectra
    for label, ct, col in [('4-spinor internal', c4_perfect, 'b'),
                             ('8-spinor internal', c8, 'r')]:
        if np.std(ct) > 0.001:
            fft = np.fft.rfft(ct - np.mean(ct))
            freqs = np.fft.rfftfreq(len(ct))
            power = np.abs(fft)**2
            ax.plot(freqs[1:len(freqs)//4], power[1:len(power)//4],
                    color=col, label=label, alpha=0.8)
    ax.set_xlabel('Frequency (1/step)')
    ax.set_ylabel('Power')
    ax.set_title('Coherence Oscillation Spectrum')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'task5_16v8_comparison.png', dpi=150)
    plt.close()
    print(f"\n  Plot saved: task5_16v8_comparison.png")

    return c4_perfect, c4_near, c8


# ============================================================================
# TASK 6: CANDIDATE CONSTANT IDENTIFICATION
# ============================================================================

def task6_candidate_constants(gamma_internal, frac_wind, results_4, c4_internal, c8_internal):
    print("\n" + "=" * 76)
    print("TASK 6: CANDIDATE CONSTANT IDENTIFICATION")
    print("  Collect all dimensionless numbers from Tasks 2-5")
    print("=" * 76)

    candidates = {}

    # 1. Internal Berry phase (from Task 3)
    candidates['gamma_internal/(2pi)'] = {
        'value': frac_wind,
        'source': 'Task 3: Berry phase of internal Coxeter cycle',
        'geometric': True,
    }
    candidates['gamma_internal/pi'] = {
        'value': gamma_internal / np.pi,
        'source': 'Task 3: Berry phase / pi',
        'geometric': True,
    }

    # 2. Coherence oscillation properties (from Task 2/5)
    for label, ct_arr in [('|0> simple', results_4.get('|0> simple', {}).get('C', None)),
                            ('4-spinor internal', c4_internal)]:
        if ct_arr is not None and np.std(ct_arr) > 0.001:
            fft = np.fft.rfft(ct_arr - np.mean(ct_arr))
            freqs = np.fft.rfftfreq(len(ct_arr))
            power = np.abs(fft)**2
            peak_idx = np.argmax(power[1:]) + 1
            peak_freq = freqs[peak_idx]
            peak_period = 1/peak_freq if peak_freq > 0 else 0
            candidates[f'period/Coxeter ({label})'] = {
                'value': peak_period / COXETER_H if COXETER_H > 0 else 0,
                'source': f'Task 5: coherence oscillation period / Coxeter number',
                'geometric': True,
            }
            candidates[f'peak_freq ({label})'] = {
                'value': peak_freq,
                'source': f'Task 5: dominant frequency of C(t)',
                'geometric': True,
            }

    # 3. Mean coherence magnitude under internal dynamics
    for label, ct_arr in [('4-spinor', c4_internal), ('8-spinor', c8_internal)]:
        if ct_arr is not None:
            mean_abs = np.mean(np.abs(ct_arr))
            std_c = np.std(ct_arr)
            candidates[f'|C|_mean ({label})'] = {
                'value': mean_abs,
                'source': f'Task 5: mean |C(t)| under internal dynamics',
                'geometric': True,
            }

    # 4. Dimensional ratios
    candidates['16/8 (tesseract/dual-tet)'] = {
        'value': 16/8,
        'source': 'Task 7: vertex ratio',
        'geometric': True,
    }
    candidates['128/16 (octeract/tesseract)'] = {
        'value': 128/16,
        'source': 'Task 7: vertex ratio',
        'geometric': True,
    }
    candidates['4D/3D'] = {
        'value': 4/3,
        'source': 'Dimension ratio tesseract/dual-tetrahedra',
        'geometric': True,
    }

    # 5. Tesseract-specific geometric numbers
    # Tesseract has 16 vertices, 32 edges, 24 faces, 8 cells
    candidates['edges/vertices'] = {
        'value': 32/16,
        'source': 'Tesseract: 32 edges / 16 vertices = 2',
        'geometric': True,
    }
    candidates['faces/edges'] = {
        'value': 24/32,
        'source': 'Tesseract: 24 faces / 32 edges = 3/4',
        'geometric': True,
    }
    candidates['cells/faces'] = {
        'value': 8/24,
        'source': 'Tesseract: 8 cells / 24 faces = 1/3',
        'geometric': True,
    }
    candidates['Euler char ratio'] = {
        'value': (16 - 32 + 24 - 8),  # = 0
        'source': 'Tesseract Euler characteristic: V-E+F-C',
        'geometric': True,
    }

    # 6. Self-duality measure
    # The tesseract dual (16-cell) has 8 vertices, 24 edges, 32 faces, 16 cells
    # The ratio of corresponding elements:
    candidates['dual_vertex_ratio'] = {
        'value': 8/16,
        'source': 'Tesseract vs dual (16-cell): 8/16 vertices',
        'geometric': True,
    }

    # 7. Berry phase from existing simulation at |0>
    # From the original: gamma = -5.7719 rad = -1.837238*pi
    gamma_ext = -5.771855
    candidates['gamma_ext/(2pi)'] = {
        'value': gamma_ext / (2 * np.pi),
        'source': 'Existing simulation: Berry phase at |0> with full drive',
        'geometric': False,  # depends on drive parameters
    }

    # 8. Ratio of internal to external Berry phase
    if abs(gamma_ext) > 1e-10:
        candidates['gamma_int/gamma_ext'] = {
            'value': gamma_internal / gamma_ext,
            'source': 'Ratio of internal to external Berry phase',
            'geometric': True,
        }

    # Print all candidates
    print(f"\n  {'#':<4}  {'Name':<35}  {'Value':<14}  {'Geometric?':<12}  {'Source'}")
    print(f"  {'-'*4}  {'-'*35}  {'-'*14}  {'-'*12}  {'-'*40}")
    for i, (name, info) in enumerate(candidates.items()):
        geo = 'YES' if info['geometric'] else 'no'
        print(f"  {i+1:<4}  {name:<35}  {info['value']:<14.6f}  {geo:<12}  {info['source'][:40]}")

    # Test each against known constants
    print(f"\n  --- Testing candidates against known constants ---")
    known = {
        '1/137': 1/137.036,
        '4/3': 4/3,
        '1/2': 1/2,
        '1/4': 1/4,
        'pi/12': np.pi/12,
        '1/3': 1/3,
        '3/4': 3/4,
        '2': 2.0,
        '8': 8.0,
        'pi/4': np.pi/4,
        '1/sqrt(2)': 1/np.sqrt(2),
    }

    print(f"\n  {'Candidate':<35}  {'Value':<12}  {'Closest known':<15}  {'Difference':<12}  {'Match?'}")
    print(f"  {'-'*35}  {'-'*12}  {'-'*15}  {'-'*12}  {'-'*8}")

    strong_candidates = []
    for name, info in candidates.items():
        val = info['value']
        if abs(val) < 1e-10:
            continue
        best_match = None
        best_diff = float('inf')
        for kname, kval in known.items():
            if abs(kval) > 1e-10:
                diff = abs(val - kval)
                rel_diff = diff / abs(kval)
                if rel_diff < best_diff:
                    best_diff = rel_diff
                    best_match = kname
                    abs_diff = diff

        match = "EXACT" if best_diff < 0.001 else ("CLOSE" if best_diff < 0.05 else "no")
        if best_match:
            print(f"  {name:<35}  {val:<12.6f}  {best_match:<15}  {abs_diff:<12.6f}  {match}")
            if match in ("EXACT", "CLOSE") and info['geometric']:
                strong_candidates.append((name, val, best_match, abs_diff))

    if strong_candidates:
        print(f"\n  STRONG CANDIDATES (geometric + matches known constant):")
        for name, val, match, diff in strong_candidates:
            print(f"    {name}: {val:.6f} ~ {match} (diff={diff:.6f})")

    return candidates


# ============================================================================
# TASK 7: OCTERACT CONNECTION AND DIMENSIONAL LADDER
# ============================================================================

def task7_octeract_connection(c4_internal, c8_internal):
    print("\n" + "=" * 76)
    print("TASK 7: OCTERACT CONNECTION AND DIMENSIONAL LADDER")
    print("  Does self-sustaining property strengthen at higher dimensions?")
    print("=" * 76)

    # Dimensional ladder
    print(f"\n  DIMENSIONAL LADDER:")
    print(f"  {'Structure':<20}  {'Dim':<6}  {'Vertices':<10}  {'Step ratio':<12}  {'Self-dual?':<12}")
    print(f"  {'-'*20}  {'-'*6}  {'-'*10}  {'-'*12}  {'-'*12}")
    structures = [
        ('Dual tetrahedra', 3, 8, '-', 'No (tet dual=tet)'),
        ('Tesseract', 4, 16, '16/8=2', 'No (dual=16-cell)'),
        ('Octeract', 7, 128, '128/16=8', 'No (dual=7-orthoplex)'),
    ]
    for name, dim, verts, ratio, sd in structures:
        print(f"  {name:<20}  {dim:<6}  {verts:<10}  {ratio:<12}  {sd:<12}")

    # Vertex ratios
    print(f"\n  Step ratios:")
    print(f"    8 -> 16:  factor {16/8} = 2 = 2^1")
    print(f"    16 -> 128: factor {128/16} = 8 = 2^3")
    print(f"    8 -> 128:  factor {128/8} = 16 = 2^4")
    print(f"    Pattern: 2^n vertices at dimension n+1 (hypercube)")
    print(f"    2^3=8, 2^4=16, 2^7=128")
    print(f"    Dimension jumps: 3->4 (step 1), 4->7 (step 3)")
    print(f"    These are the division algebra dimensions: C(2), H(4), O(8)")
    print(f"    Jump pattern: 1, 3 -> next would be 7 (sedenions, dim 15, 2^15=32768)")

    # Compare coherence behaviour
    n_test = 300
    coupling_strengths = [0.3, 0.5, 1.0]

    print(f"\n  COHERENCE COMPARISON (standing wave, internal dynamics, {n_test} steps):")
    print(f"  {'System':<20}  {'coupling':<10}  {'|C|_mean':<10}  {'std(C)':<10}  "
          f"{'max|C|':<10}  {'Oscillates?':<15}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*15}")

    for coupling in coupling_strengths:
        # 4-spinor
        s4 = Merkabit4([1, 0, 0, 0], [0, 0, 0, 1])
        c4 = [s4.coherence]
        for step in range(n_test):
            s4 = internal_step_4(s4, step % COXETER_H, coupling)
            c4.append(s4.coherence)
        c4 = np.array(c4)

        osc4 = "YES" if np.std(c4) > 0.05 else "no"
        print(f"  {'4-spinor':<20}  {coupling:<10.1f}  {np.mean(np.abs(c4)):<10.4f}  "
              f"{np.std(c4):<10.4f}  {np.max(np.abs(c4)):<10.4f}  {osc4:<15}")

        # 8-spinor
        s8 = Merkabit8([1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1])
        c8 = [s8.coherence]
        for step in range(n_test):
            s8 = internal_step_8(s8, step % COXETER_H, coupling)
            c8.append(s8.coherence)
        c8 = np.array(c8)

        osc8 = "YES" if np.std(c8) > 0.05 else "no"
        print(f"  {'8-spinor':<20}  {coupling:<10.1f}  {np.mean(np.abs(c8)):<10.4f}  "
              f"{np.std(c8):<10.4f}  {np.max(np.abs(c8)):<10.4f}  {osc8:<15}")

    # Self-sustaining test: does higher dimension maintain coherence better?
    print(f"\n  SELF-SUSTAINING STRENGTH vs DIMENSION:")
    # Measure: start near |0>, perturb, evolve, check if coherence stays bounded
    n_trials = 1000
    n_evolve = 200
    eps = 0.05

    dims_data = []
    for dim, cls, step_fn in [(2, Merkabit2, None),
                                (4, Merkabit4, internal_step_4),
                                (8, Merkabit8, internal_step_8)]:
        bounded_count = 0
        mean_amplitudes = []
        for trial in range(n_trials):
            u = np.zeros(dim, dtype=complex)
            u[0] = 1.0
            v = np.zeros(dim, dtype=complex)
            v[-1] = np.sqrt(1 - eps**2)
            v[0] = eps * np.exp(1j * 2 * np.pi * trial / n_trials)

            if dim == 2:
                s = Merkabit2(u, v)
                # No internal dynamics -> frozen
                mean_amplitudes.append(abs(s.coherence))
                if abs(s.coherence) < 0.3:
                    bounded_count += 1
            elif dim == 4:
                s = Merkabit4(u, v)
                cs = [abs(s.coherence)]
                for step in range(n_evolve):
                    s = internal_step_4(s, step % COXETER_H)
                    cs.append(abs(s.coherence))
                mean_amplitudes.append(np.mean(cs))
                if np.max(cs) < 0.3:
                    bounded_count += 1
            else:
                s = Merkabit8(u, v)
                cs = [abs(s.coherence)]
                for step in range(n_evolve):
                    s = internal_step_8(s, step % COXETER_H)
                    cs.append(abs(s.coherence))
                mean_amplitudes.append(np.mean(cs))
                if np.max(cs) < 0.3:
                    bounded_count += 1

        frac_bounded = bounded_count / n_trials
        mean_amp = np.mean(mean_amplitudes)
        dims_data.append((dim, frac_bounded, mean_amp))
        print(f"  dim={dim:>2}: bounded fraction = {frac_bounded:.4f}, "
              f"mean|C| = {mean_amp:.4f}")

    # Ratio analysis
    if len(dims_data) >= 2:
        print(f"\n  Ratio analysis:")
        for i in range(1, len(dims_data)):
            d1, f1, a1 = dims_data[i-1]
            d2, f2, a2 = dims_data[i]
            ratio_f = f2/f1 if f1 > 0 else float('inf')
            ratio_a = a2/a1 if a1 > 0 else float('inf')
            print(f"    dim {d1}->{d2}: bounded ratio = {ratio_f:.4f}, "
                  f"amplitude ratio = {ratio_a:.4f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 76)
    print("  TESSERACT ZERO POINT CONSTANT INVESTIGATION")
    print("  Finding the geometric measure of self-sustaining coherence")
    print("=" * 76)
    print(f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    t0 = time.time()

    # Task 2
    results_4, c2_t, c8_t = task2_self_sustaining_coherence()

    # Task 3
    gamma_internal, frac_wind = task3_internal_berry_phase()

    # Task 4
    basin_4, basin_8 = task4_basin_geometry()

    # Task 5
    c4_internal, c4_near, c8_internal = task5_comparison()

    # Task 6
    candidates = task6_candidate_constants(gamma_internal, frac_wind,
                                            results_4, c4_internal, c8_internal)

    # Task 7
    task7_octeract_connection(c4_internal, c8_internal)

    # =====================================================================
    # FINAL SYNTHESIS
    # =====================================================================
    print("\n" + "=" * 76)
    print("  FINAL SYNTHESIS: THE ZERO POINT CONSTANT")
    print("=" * 76)

    print(f"\n  INTERNAL BERRY PHASE:")
    print(f"    gamma_internal = {gamma_internal:.6f} rad")
    print(f"    gamma/(2pi) = {frac_wind:.6f}")
    print(f"    gamma/pi = {gamma_internal/np.pi:.6f}")

    print(f"\n  COHERENCE UNDER INTERNAL DYNAMICS:")
    print(f"    4-spinor |C|_mean = {np.mean(np.abs(c4_internal)):.6f}")
    print(f"    8-spinor |C|_mean = {np.mean(np.abs(c8_internal)):.6f}")

    # The key question: what dimensionless number characterizes
    # self-sustaining coherence in the tesseract?
    print(f"\n  KEY DIMENSIONLESS NUMBERS:")
    print(f"    1. Faces/Edges ratio (tesseract): 24/32 = {24/32} = 3/4")
    print(f"    2. Cells/Faces ratio (tesseract): 8/24 = {8/24:.6f} = 1/3")
    print(f"    3. Dimension ratio (4D/3D): {4/3:.6f} = 4/3")
    print(f"    4. Euler characteristic: V-E+F-C = {16-32+24-8} = 0 (closed)")
    print(f"    5. Internal Berry phase / (2pi) = {frac_wind:.6f}")
    print(f"    6. Internal/External Berry ratio = {gamma_internal/(-5.771855):.6f}")

    # Compute the tesseract self-duality defect
    # Tesseract: (16, 32, 24, 8)
    # 16-cell:   (8, 24, 32, 16)
    # Self-dual would mean these are equal
    # Defect = sum of |tess_i - dual_i| / sum of tess_i
    tess = np.array([16, 32, 24, 8])
    dual = np.array([8, 24, 32, 16])
    defect = np.sum(np.abs(tess - dual)) / np.sum(tess)
    print(f"    7. Self-duality defect: {defect:.6f}")
    # Note: sum |tess_i - dual_i| = |16-8|+|32-24|+|24-32|+|8-16| = 8+8+8+8 = 32
    # sum tess_i = 80
    # defect = 32/80 = 0.4 = 2/5
    print(f"       = 32/80 = 2/5 = 0.4")

    # The 24-cell IS self-dual in 4D
    # 24-cell: (24, 96, 96, 24)
    cell24 = np.array([24, 96, 96, 24])
    print(f"    8. 24-cell self-duality defect: {np.sum(np.abs(cell24 - cell24[::-1]))/np.sum(cell24):.6f} (ZERO - truly self-dual)")
    print(f"       24-cell has {24} vertices, {96} edges")

    # The ratio tesseract_vertices / 24-cell_vertices = 16/24 = 2/3
    print(f"    9. tesseract/24-cell vertices: {16/24:.6f} = 2/3")

    # KEY INSIGHT: the 24-cell, not the tesseract, is the self-dual 4D polytope
    # But the tesseract's COUNTER-ROTATING pair (tesseract + 16-cell) together
    # have 16+8 = 24 vertices... which IS the 24-cell!
    print(f"\n  CRITICAL INSIGHT:")
    print(f"    Tesseract (16 vertices) + its dual 16-cell (8 vertices) = 24 vertices")
    print(f"    This is the vertex count of the 24-CELL, the self-dual 4D polytope!")
    print(f"    The counter-rotating pair (u on tesseract, v on 16-cell)")
    print(f"    together form the self-dual 24-cell structure.")
    print(f"\n    The zero point constant candidates from this structure:")
    print(f"      - 24-cell self-duality: defect = 0 (exact closure)")
    print(f"      - Vertex ratio: tesseract/combined = 16/24 = 2/3")
    print(f"      - Dual/(dual+tess) = 8/24 = 1/3")
    print(f"      - Combined/tesseract = 24/16 = 3/2")

    # The internal Berry phase and what it means
    print(f"\n  RECOMMENDED NEXT SIMULATION:")
    print(f"    1. Build the 24-cell merkabit explicitly (24 vertices, self-dual)")
    print(f"    2. Compute Berry phase of the 24-cell's internal ouroboros loop")
    print(f"    3. Test whether 24-cell coherence is truly self-sustaining")
    print(f"    4. The zero point constant should emerge from the 24-cell geometry")
    print(f"    5. Test: does gamma_24cell/(2pi) = a clean fraction?")

    print(f"\n  Total computation time: {time.time() - t0:.1f} seconds")
    print("=" * 76)


if __name__ == "__main__":
    main()
