"""Added after the runs: is the gap at tripled insulin in the estimate or in its installation? Each
estimate's implied shift (Eq. 2) at x2 and x3 on the 600 held-out stays, over the simulator's true shift at the same
factor on the same stay-hours; beside the edit's held-out readout at x3 (results/generalize). If an estimate implies about
the truth at x3 while its edit falls short, the x3 gap lies in installation; if the estimate itself falls short, in the
estimate. CPU, synthetic data.

    python scripts/x3_identification.py   -> results/x3_identification.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "agent")]
import audit_eq2 as ae  # noqa: E402
import decompose_same_stays as dss  # noqa: E402
import scripted_estimate as se  # noqa: E402
from backend import sim  # noqa: E402


def claude_kernel_sized():
    """The script with Claude's estimate installs Claude's rounded kernel sized to Claude's own tuning
    readout, as scripts/decompose_same_stays.py scales it (results/decompose_same_stays.json); the same scale here"""
    k = np.r_[[-2.0, -2.3, -2.0, -1.6, -1.1, -0.9, -0.6, -0.4], np.zeros(4)]
    sb = json.loads((ROOT / "results" / "scripted_baseline.json").read_text())
    target = sb["scripted install, Claude's estimate"]["tuning"]["targets"]["insulin_x2"]
    tun = ae.tuning_panel(ROOT / "agent_data" / "e03" / "data" / "tuning.csv")
    return k * target / float(se.implied(tun, "I", "glucose_finger", k, 2.0)[0])


def truth(f: float, seed: int = 303, n: int = 600) -> float:
    """audit_eq2.scored_truth at factor f: same stays, hours and pooling"""
    import fresh_null_seeds as fns
    s = sim.simulate(ae.MAIN, n, seed)
    p = fns.panel_from_sim(s)
    r = sim.replay(ae.MAIN, n, seed, s, {"INS": f}, 12)
    d = r["edited"]["G"][:, sim.PRE:sim.PRE + sim.HOURS] - r["factual"]["G"][:, sim.PRE:sim.PRE + sim.HOURS]
    R, y, vals = p["I"], p["glucose_finger"], []
    for t in range(13, 25):
        ok = np.isfinite(y[:, t]) & (R[:, 12:t] > 0).any(axis=1)
        vals.append(d[ok, t])
    return float(np.concatenate(vals).mean())


hel = ae.tuning_panel(dss.held_out_dir() / "tuning.csv")
T = {f: truth(f) for f in (2.0, 3.0)}
kern = {"script": np.r_[json.loads((ROOT / "results" / "scripted" / "estimate_e03.json").read_text())["pairs"]["I->glucose_finger"]["psi"][:8], np.zeros(4)],
        "script_claude_estimate": claude_kernel_sized()}
out = {"truth": T, "runs": {}}
for run in ("claude_a", "claude_d", "claude_e", "openai_a", "openai_d", "openai_e", "script", "script_claude_estimate"):
    psi = kern[run] if run in kern else dss.profile(run)
    imp = {f: float(se.implied(hel, "I", "glucose_finger", psi, f)[0]) for f in T}
    out["runs"][run] = {"implied_over_truth_x2": imp[2.0] / T[2.0], "implied_over_truth_x3": imp[3.0] / T[3.0]}
    print(run, {k: round(v, 3) for k, v in out["runs"][run].items()}, flush=True)
(ROOT / "results" / "x3_identification.json").write_text(json.dumps(out, indent=1) + "\n")
