"""Tests for the results corrected in the revision (referee report, Oct 2026)."""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import kernel
import partial_id as PI


def lyapunov_mass_kernel(A, B, h='1e-12'):
    """-d ln lambda / d gamma at the photon orbit (r A' = 2A), lambda^2 = B (2A - r^2 A'')/(2 r^2), in 40-digit
    arithmetic (M = 1)."""
    import mpmath as mp
    mp.mp.dps = 40
    def lam(gm):
        Ag = lambda r: A(r, gm)
        rc = mp.findroot(lambda r: r * mp.diff(Ag, r) - 2 * Ag(r), 3)
        return mp.sqrt(B(rc, gm) * (2 * Ag(rc) - rc ** 2 * mp.diff(Ag, rc, 2)) / (2 * rc ** 2))
    h = mp.mpf(h)
    return float(-(mp.log(lam(h)) - mp.log(lam(-h))) / (2 * h))


@pytest.mark.parametrize('n', [2, 3, 4])
def test_damping_kernel_linked_family(n):
    A = lambda r, g: 1 - 2 / r + g * r ** -n
    assert lyapunov_mass_kernel(A, A) == pytest.approx(kernel(n, 'eikonal_wI'), abs=2e-5)


@pytest.mark.parametrize('n', [2, 3])
def test_damping_kernel_radial_families(n):
    S = lambda r, g: 1 - 2 / r
    add = lambda r, g: 1 - 2 / r + g * r ** -n                       # referee's additive form (irregular horizon)
    mult = lambda r, g: (1 - 2 / r) * (1 + g * r ** -n)              # regular completion used in Table I
    assert lyapunov_mass_kernel(S, add) == pytest.approx(-0.5 * 3.0 ** (1 - n), abs=2e-5)
    assert lyapunov_mass_kernel(S, mult) == pytest.approx(kernel(n, 'grr_wI'), abs=2e-5)


def test_referee_counterexample_and_identified_set():
    D = 1e-3 * PI.shell_resp(4.0) + 1e-3 * PI.shell_resp(10.0)
    Delta = PI.deficit_template(2)
    for t in np.linspace(-0.004, 0.003, 9):
        m1, m2 = 1e-3 + t / 4, 1e-3 - t / 3
        assert m1 >= -1e-15 and m2 >= -1e-15
        assert np.allclose(t * Delta + m1 * PI.shell_resp(4.0) + m2 * PI.shell_resp(10.0), D, atol=1e-18)
    lo, hi = PI.identified_set(D, 2)
    assert lo == pytest.approx(8 * (D[1] - D[0])) and hi == pytest.approx(24 / 7 * D[1])


def test_point_identification_with_stable_orbits():
    D = 0.02 * np.ones(2) + 0.01 * PI.deficit_template(2)
    lo, hi = PI.identified_set(D, 2, Rmin=6.0)
    assert lo == pytest.approx(0.01) and hi == pytest.approx(0.01)


def test_constrained_profile_is_asymmetric():
    grid = np.array([-0.05, 0.0, 0.05]); sig = [0.01, 0.01]
    p_env = PI.profile(np.zeros(2), 2, sig, 'R>=6M', grid)[[0, 2]]
    p0 = PI.profile(np.zeros(2), 2, sig, 'none', grid)[[0, 2]]
    a = PI.deficit_template(2); s = np.ones(2)
    c2 = (a @ s) ** 2 / ((a @ a) * (s @ s))
    assert p_env[1] / p0[1] == pytest.approx(1.0, rel=1e-9)          # delta > 0: no inflation
    assert p_env[0] / p0[0] == pytest.approx(1 - c2, rel=1e-9)        # delta < 0: sin^2 phi


def test_kantorovich_bound_needs_exact_anchor():
    k, s = np.array([7 / 4, 1.0]), np.ones(2)
    def inv_sin(a):
        Ci = np.linalg.inv(np.diag([7 / 4, 1.0]) + a * np.ones((2, 2)))
        c2 = (k @ Ci @ s) ** 2 / ((k @ Ci @ k) * (s @ Ci @ s))
        return 1 / np.sqrt(1 - c2)
    assert inv_sin(0.0) == pytest.approx(11 / 3)
    assert inv_sin(0.01) == pytest.approx(3.6403, abs=1e-4) and inv_sin(0.01) < 11 / 3


@pytest.mark.parametrize('n,ref', [(2, -0.0973441851), (3, -0.0154781798), (4, -0.0022392505)])
def test_flare_kernel_same_estimator(n, ref):
    from flare_consistent import flare_kernel
    assert flare_kernel(n) == pytest.approx(ref, abs=2e-9)


def test_isco_shadow_gap_n6():
    a = PI.deficit_template(6)
    assert a[1] - a[0] == pytest.approx(1 / 10368, rel=1e-12)
