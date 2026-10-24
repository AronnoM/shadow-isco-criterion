"""Recompute data/do2019_chain_summary.json from the Keck nested-sampling chains (Do et al. 2019, Data S3).

The chains are not redistributed here. Download aav8137_data_s3.zip from the Science supplement
(https://www.science.org/doi/10.1126/science.aav8137), extract one chain as a whitespace- or comma-separated
table, and pass the column indices of the weight, the mass (10^6 Msun) and R0 (kpc):

    python3 scripts/chain_summary.py CHAIN_FILE --w 0 --M 2 --R0 3

The stored summary was computed with the nested-sampling weights; this script reproduces the same statistics.
"""
import argparse, json
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('chain'); p.add_argument('--w', type=int, required=True)
p.add_argument('--M', type=int, required=True); p.add_argument('--R0', type=int, required=True)
p.add_argument('--delimiter', default=None)
a = p.parse_args()
X = np.loadtxt(a.chain, delimiter=a.delimiter, comments='#')
w = X[:, a.w] / X[:, a.w].sum(); M = X[:, a.M]; R = X[:, a.R0]
mean = lambda x: float(np.sum(w * x))
sd = lambda x: float(np.sqrt(np.sum(w * (x - mean(x)) ** 2)))
lM, lR = np.log(M), np.log(R)
cov = float(np.sum(w * (lM - mean(lM)) * (lR - mean(lR))))
th = lM - lR                                          # ln theta_g up to a constant
z = (th - mean(th)) / sd(th)
out = dict(n_samples=len(w), n_eff=float(1 / np.sum(w ** 2)), mean_M=mean(M), sd_M=sd(M), mean_R0_kpc=mean(R),
           sd_R0_kpc=sd(R), mean_lnM=mean(lM), sd_lnM=sd(lM), mean_lnR0=mean(lR), sd_lnR0=sd(lR), cov_lnM_lnR0=cov,
           corr_ln=cov / (sd(lM) * sd(lR)), lnTheta_skew=float(np.sum(w * z ** 3)), lnTheta_exkurt=float(np.sum(w * z ** 4) - 3))
print(json.dumps(out, indent=1))
