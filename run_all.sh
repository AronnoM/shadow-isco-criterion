#!/bin/sh
# Reproduce every number, table and figure of the paper. Run from the repository root.
# Runtime: about 45 minutes in total; s2_refit.py (~20 min), sgra_real.py (~10 min) and s2_deform.py (~5 min)
# dominate. Their outputs are already stored in results/, so individual steps can be skipped.
set -e
cd "$(dirname "$0")"
echo "=== unit tests"; python3 -m pytest -q tests/
for s in verify_decomposition attack1_structure attack1b_correlated_fix attack2_physics attack3_statistics \
         p2_profiling bias_kernels grr_blindness psaltis_check thin_shell env_theorem env_monotone \
         env_conditional_theorem theorem_audit partial_id flare_consistent exact_mimic \
         qnm_spectral qnm_validate s2_refit s2_fsp s2_deform s2_photon s2_jackknife sgra_real paper_figures; do
  echo "=== $s"; python3 "src/$s.py"
done
