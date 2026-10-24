"""Validate the WKB3 ringdown kernels with direct pseudo-spectral QNMs."""
import json
import numpy as np
from qnm_spectral import fundamental
from wkb import qnm_wkb3
from core import kernel

out = []
for n in [2, 3]:
    for s in [0, 1, 2]:
        for l in [2, 3, 4]:
            w0, _ = fundamental(0.0, n, l, s)
            ker = {}
            for h in [2e-3, 1e-3]:
                wp, _ = fundamental(h, n, l, s, guess=w0); wm, _ = fundamental(-h, n, l, s, guess=w0)
                ker[h] = (-(np.log(wp.real) - np.log(wm.real)) / (2 * h), -(np.log(-wp.imag) - np.log(-wm.imag)) / (2 * h))
            bR = (4 * ker[1e-3][0] - ker[2e-3][0]) / 3          # Richardson
            bI = (4 * ker[1e-3][1] - ker[2e-3][1]) / 3
            e = 1e-4
            wpw, wmw = qnm_wkb3(e, n=n, l=l, s=s), qnm_wkb3(-e, n=n, l=l, s=s)
            bR_w = -(np.log(wpw.real) - np.log(wmw.real)) / (2 * e)
            ksh = kernel(n, 'shadow')
            row = dict(n=n, s=s, l=l, w0=[w0.real, w0.imag], beta_R=bR, beta_I=bI, beta_R_wkb3=bR_w,
                       beta_shadow=ksh, leak_pct=100 * (bR - ksh) / abs(ksh), richardson_change=abs(bR - ker[1e-3][0]))
            out.append(row)
            print(f"n={n} s={s} l={l}: beta_R spectral {bR:+.6f} | WKB3 {bR_w:+.6f} | shadow {ksh:+.6f} |"
                  f" leak {row['leak_pct']:+.2f}% | beta_I {bI:+.5f}")
json.dump(out, open('results/qnm_validate.json', 'w'), indent=1, default=float)
