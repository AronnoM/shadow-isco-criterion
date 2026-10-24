"""Closest approach to Sgr A* of the light from S0-2 over the observed epochs (referee point 5: Birkhoff's theorem
covers the stellar motion, not photons). A ray from the star toward the observer (-Z) has closest approach equal to
the projected separation if the star is behind the black hole (Z > 0), and to the stellar radius otherwise."""
import json
import numpy as np
from scipy.integrate import solve_ivp
import s2_refit as S

th = np.array(json.load(open('results/s2_deform.json'))['do_noise']['theta'])
GM = S.GM1 * th[0] * 1e6
s0 = S.initial_state(th[0] * 1e6, *th[7:13])


def state(t):
    if t == th[8]:
        return s0
    return solve_ivp(S.rhs, (th[8], t), s0, method='DOP853', rtol=1e-11, atol=1e-9, args=(GM, 0.0, np.inf)).y[:, -1]


def closest(ts):
    s = np.array([state(t) for t in ts]).T
    x, y, z = s[:3]
    return np.where(z > 0, np.hypot(x, y), np.sqrt(x * x + y * y + z * z)), np.sqrt(x * x + y * y + z * z)


if __name__ == '__main__':
    AU_in_M = 23.56 * 4.3 / th[0]
    b_obs, _ = closest(np.concatenate([S.t_ast, S.t_rv]))
    t_all = np.linspace(th[8] - 8.0, th[8] + 8.0, 1601)
    b_all, r_all = closest(t_all)
    out = dict(r_peri_AU=float(r_all.min()), b_min_orbit_AU=float(b_all.min()), b_min_obs_AU=float(b_obs.min()),
               AU_in_M=AU_in_M)
    print(f"pericentre {out['r_peri_AU']:.1f} AU ({out['r_peri_AU']*AU_in_M:.0f} M); light closest approach: "
          f"{out['b_min_obs_AU']:.1f} AU ({out['b_min_obs_AU']*AU_in_M:.0f} M) over observed epochs")
    json.dump(out, open('results/s2_photon.json', 'w'), indent=1)
