# Input data

| File | Content | Source |
| --- | --- | --- |
| `do2019_astrometry.csv` | S0-2 astrometry, 46 rows as released (the analysis drops 2018.673, as Do et al. do) | Do et al. 2019, Science 365, 664, Data S2 |
| `do2019_rv.csv` | S0-2 radial velocities; the analysis uses the 115 measurements of 2000–2018 | Do et al. 2019, Data S1 |
| `do2019_chain_summary.json` | weighted means, covariances and moments of the GR and Υ-free nested-sampling chains | computed from Do et al. 2019, Data S3 (not redistributed; see `scripts/chain_summary.py`) |

GRAVITY and EHT inputs are a handful of published numbers and are set in `src/sgra_real.py`:
GRAVITY Collaboration 2022, A&A 657, L12 (M = 4.297 ± 0.012 × 10⁶ M☉, R₀ = 8277 ± 9 pc, systematics ≈ 0.040 and ≈ 30 pc;
ρ(M, R₀) ≈ 0.96 digitized from their corner plot, 0.95–0.97 scanned), and EHT Collaboration 2022, ApJL 930, L12
(θ_g = 4.8 +1.4/−0.7 μas). This radial-velocity file also contains one 2019 measurement outside Do et al.'s
2000–2018 set; `src/s2_refit.py` drops it.
