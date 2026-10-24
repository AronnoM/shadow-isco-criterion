# shadow-isco-criterion

Code and data for

> A. Mridha, *Can static matter shells imitate a deformed black hole? A shadow–ISCO criterion for multi-probe mass tests* (2026).

Every number, table and figure in the paper is produced by the scripts here. All outputs are already stored in
`results/` and `paper/figures/`, so you can check numbers without rerunning anything.

## Quick start

```sh
pip install -r requirements.txt
python3 -m pytest -q tests/        # 85 tests, a few seconds
./run_all.sh                       # everything, about 45 minutes
python3 src/partial_id.py          # or any single script, run from the repository root
```

`src/paper_figures.py` needs LaTeX with cm-super (figures use LaTeX text).

## Where each result comes from

| Paper item | Script(s) | Output |
| --- | --- | --- |
| Propositions 1–7, validation, stress tests (Table V) | `core.py`, `verify_decomposition.py`, `attack1_structure.py`, `attack1b_correlated_fix.py`, `attack2_physics.py`, `attack3_statistics.py`, `p2_profiling.py` | `results/attack*.json`, `results/p2_profiling.json` |
| Table I (kernels) | `bias_kernels.py`, `core.py` (`kernel`), `grr_blindness.py`, `flare_consistent.py` (flare rows) | `results/flare_consistent.json` |
| Shadow kernel vs. Psaltis et al. 2021 | `psaltis_check.py` | `results/psaltis_check.json` |
| Thin-shell response, Eq. (13), Fig. 2 | `thin_shell.py`, `env_theorem.py` | `results/env_theorem.json` |
| Table II, Theorem 1, Corollaries 2–3, numerical checks | `env_conditional_theorem.py`, `theorem_audit.py` (independent audit), `env_monotone.py` | `results/env_conditional_theorem.json`, `results/theorem_audit.json` |
| Corollary 4, identified set, Fig. 4, finite-anchor bound | `partial_id.py` | `results/partial_id.json` |
| Sec. IV D (flare, one estimator), exact two-shell mimic | `flare_consistent.py`, `exact_mimic.py` | `results/flare_consistent.json`, `results/exact_mimic.json` |
| Sec. IV E, App. E (S0-2 refit) | `s2_refit.py`, `s2_fsp.py`, `s2_deform.py`, `s2_jackknife.py`, `s2_photon.py` | `results/s2_*.json` |
| Sec. V, Tables III–IV, Fig. 6 (Sgr A*) | `sgra_real.py` | `results/sgra_real.json`, `results/sgra_manifest.csv` (all 3456 analyses) |
| App. D, Fig. 7 (quasinormal modes) | `qnm_spectral.py`, `qnm_validate.py`, `wkb.py` | `results/qnm_validate.json` |
| Figs. 1–7 | `paper_figures.py` | `paper/figures/*.pdf` |

## Layout

```
src/        analysis code (run from the repository root)
tests/      85 unit tests, including an independent least-squares implementation of the framework
data/       Keck S0-2 data and chain summary (see data/README.md for provenance)
results/    JSON outputs of every script; sgra_manifest.csv lists every Sgr A* analysis
scripts/    chain_summary.py: recompute the chain summary from Do et al. Data S3
paper/      figures
legacy/     earlier working scripts, superseded and not used by the paper
```

## Scope

The worked physics covers static, spherically symmetric deformations of Schwarzschild with integer powers, not
rotating Kerr. The theorem concerns concentric static thin shells at first order.

## Licence and citation

The code is released under the MIT licence; the data files remain subject to the terms of their original
publications. Please cite the paper; `CITATION.cff` gives the details.
