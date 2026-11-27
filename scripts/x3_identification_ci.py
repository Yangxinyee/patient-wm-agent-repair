"""Scripts/x3_identification.py with intervals. Each estimate's implied shift (Eq. 2) at x2 and x3 over the
simulator's true shift on the same 600 held-out stays and stay-hours, per stay, then the ratio of sums with a 95% interval
resampling stays jointly (2,000 draws; the estimates fixed); beside each edit's held-out readout at x3. Points equal
x3_identification.json's. CPU, synthetic data.

    python scripts/x3_identification_ci.py   -> results/x3_identification_ci.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "agent")]
import audit_eq2 as ae  # noqa: E402
import decompose_same_stays as dss  # noqa: E402
import fresh_null_seeds as fns  # noqa: E402
import scripted_estimate as se  # noqa: E402
from backend import sim  # noqa: E402

LAGS, START, N_DRAWS = 12, 12, 2000


def claude_kernel_sized():
    """The script with Claude's estimate installs Claude's rounded kernel sized to Claude's own tuning
    readout, as scripts/decompose_same_stays.py scales it (results/decompose_same_stays.json); the same scale here"""
    k = np.r_[[-2.0, -2.3, -2.0, -1.6, -1.1, -0.9, -0.6, -0.4], np.zeros(4)]
    sb = json.loads((ROOT / "results" / "scripted_baseline.json").read_text())
    target = sb["scripted install, Claude's estimate"]["tuning"]["targets"]["insulin_x2"]
    tun = ae.tuning_panel(ROOT / "agent_data" / "e03" / "data" / "tuning.csv")
    return k * target / float(se.implied(tun, "I", "glucose_finger", k, 2.0)[0])


def masks(R, y):
    """per hour t in 13..24: stays with the outcome read at t and the drug run at an hour in [12, t)"""
    return {t: np.isfinite(y[:, t]) & (R[:, START:t] > 0).any(axis=1) for t in range(START + 1, START + 13)}


def implied_per_stay(R, y, psi, f):
    s, n = np.zeros(R.shape[0]), np.zeros(R.shape[0])
    for t, ok in masks(R, y).items():
        sh = np.zeros(R.shape[0])
        for L in range(1, LAGS + 1):
            if t - L >= START:
                sh += psi[L - 1] * (f - 1) * R[:, t - L]
        s += np.where(ok, sh, 0.0)
        n += ok
    return s, n


def truth_per_stay(f, seed=303, n=600):
    s = sim.simulate(ae.MAIN, n, seed)
    p = fns.panel_from_sim(s)
    r = sim.replay(ae.MAIN, n, seed, s, {"INS": f}, 12)
    d = r["edited"]["G"][:, sim.PRE:sim.PRE + sim.HOURS] - r["factual"]["G"][:, sim.PRE:sim.PRE + sim.HOURS]
    tot, cnt = np.zeros(n), np.zeros(n)
    for t, ok in masks(p["I"], p["glucose_finger"]).items():
        tot += np.where(ok, d[:, t], 0.0)
        cnt += ok
    return tot, cnt


def main():
    hel = ae.tuning_panel(dss.held_out_dir() / "tuning.csv")
    R, y = hel["I"], hel["glucose_finger"]
    T = {f: truth_per_stay(f) for f in (2.0, 3.0)}
    kern = {"script": np.r_[json.loads((ROOT / "results" / "scripted" / "estimate_e03.json").read_text())
                            ["pairs"]["I->glucose_finger"]["psi"][:8], np.zeros(4)],
            "script_claude_estimate": claude_kernel_sized()}
    pick = np.random.default_rng(0).integers(0, R.shape[0], size=(N_DRAWS, R.shape[0]))
    out = {"draws": N_DRAWS, "runs": {}}
    for run in ("claude_a", "claude_d", "claude_e", "openai_a", "openai_d", "openai_e", "script", "script_claude_estimate"):
        psi = kern[run] if run in kern else dss.profile(run)
        o = {}
        for f, k in ((2.0, "x2"), (3.0, "x3")):
            imp, n = implied_per_stay(R, y, psi, f)
            tt, tn = T[f]
            assert np.array_equal(n, tn), "the estimate and the truth must use the same stay-hours"
            point = imp.sum() / tt.sum()
            boot = imp[pick].sum(1) / tt[pick].sum(1)
            o[k] = [round(float(point), 4)] + [round(float(x), 4) for x in np.quantile(boot, [0.025, 0.975])]
        out["runs"][run] = o
        print(run, o, flush=True)
    (ROOT / "results" / "x3_identification_ci.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
