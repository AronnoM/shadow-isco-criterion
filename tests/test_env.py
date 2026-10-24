"""Checks for the conditional environment theorem, the S2 refit model and the Psaltis comparison."""
import os, sys
import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)

from env_conditional_theorem import inferred_masses, construct, shell_template   # noqa: E402


def test_single_shell_lapse_deficit():
    """Deficit of a lapse-type probe inside one shell, relative to the anchor, is m (R-3M)/(R-2M) to O(m^2)."""
    for R in [4.0, 10.0, 300.0]:
        m = 1e-6
        Mi = inferred_masses(1.0, [(R, m)], [('lapse', 3.0), ('anchor', None)])
        assert (Mi[1] - Mi[0]) / m == pytest.approx((R - 3) / (R - 2), rel=1e-4)


def test_probe_outside_shell_sees_total_mass():
    Mi = inferred_masses(1.0, [(5.0, 0.01)], [('lapse', 6.0), ('anchor', None)])
    assert Mi[0] == pytest.approx(Mi[1], rel=1e-14)


@pytest.mark.parametrize("seed", range(5))
def test_construction_reproduces_monotone_deficits(seed):
    rng = np.random.default_rng(seed)
    K = 3
    r = np.sort(rng.uniform(3.0, 60.0, K))
    Delta = np.sort(rng.uniform(0.0, 0.4, K))[::-1]
    delta = 1e-6
    shells = construct(r, Delta, delta)
    assert all(m >= 0 for _, m in shells)
    Mi = inferred_masses(1.0, shells, [('lapse', x) for x in r] + [('anchor', None)])
    assert np.max(np.abs((Mi[-1] - Mi[:-1]) / delta - Delta)) < 1e-4


def test_step_templates_are_non_increasing():
    r = np.array([3.0, 6.0, 9.0])
    for R in np.linspace(3.01, 50, 40):
        t = shell_template(r, R)
        assert np.all(t >= 0) and np.all(np.diff(t) <= 0)


def test_increasing_deficits_not_reachable():
    """Shadow + ISCO-frequency deficits for n = 2 increase outward: no non-negative step combination fits."""
    from scipy.optimize import nnls
    from core import kernel
    r = np.array([3.0, 6.0])
    D = np.array([-kernel(2, 'shadow'), -kernel(2, 'isco_freq')])
    grid = np.linspace(3.01, 3000, 3000)
    S = np.column_stack([shell_template(r, R) for R in grid])
    _, res = nnls(S, D)
    assert res / np.linalg.norm(D) > 0.1


def test_s2_inside_pericentre_shell_is_exactly_degenerate():
    import s2_refit as S
    th = np.array([4.0, 7.99, -1.49, -0.63, 0.085, 0.214, -9.4, 15.395, 2018.38, 0.8826, 133.96, 66.45, 227.70, -78.6])
    a = S.residuals(th)
    th2 = th.copy(); th2[0] = th[0] / 1.05
    b = S.residuals(th2, 0.05, 50.0)                 # 5% of M in a shell at 50 AU (pericentre ~115 AU)
    assert np.max(np.abs(a - b)) < 1e-6


def test_psaltis_eq9_coefficients():
    from scipy.optimize import minimize_scalar
    eps = 1e-6
    b0 = 3 * np.sqrt(3)
    for i, c in [(1, -1 / 3), (2, 1 / 9), (3, -1 / 27)]:
        A = lambda r: 1 - 2 / r - 2 * (-1) ** i * eps / r ** (i + 1)
        b = minimize_scalar(lambda r: r / np.sqrt(A(r)), bounds=(2.5, 4), method='bounded',
                            options={'xatol': 1e-12}).fun
        assert (b / b0 - 1) / eps == pytest.approx(c, rel=1e-3)
