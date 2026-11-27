"""How the training records fill the glucose tiers, in main and in no-effect worlds.

The estimator of Claude's first main-world run controls for the finger-stick reading by tiers cut at 150, 180, 220 and
250 mg/dL; the sliding scale steps at 150, 180, 220, 260 and 300. With inert insulin glucose is never lowered, so the
no-effect worlds may hold more stay-hours in the top tier. For each world's 30,000 training stays (the records the
estimators fit on) this counts
  readings  every finger-stick reading, by bin
  doses     every stay-hour with insulin running, by the latest finger-stick at or before that hour, with the mean rate
bins: <150, 150-180, 180-220, 220-250, 250-260, 260-300, >=300 mg/dL (Claude's top tier is >=250: the last three).

Worlds as in scripts/audit_eq2.py: main e03 (training seed 101) and e5101-e5105; no-effect n03 (seed 101) and
n4101-n4105. Records are the simulator's, as the agents read them (audit_eq2.check: e03 here equals the agents' file).

    python scripts/tier_occupancy.py   -> results/tier_occupancy.json      (CPU, seconds; synthetic data only)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "scripts"))
from backend import sim  # noqa: E402  (the simulator, called and never changed)
import fresh_null_seeds as fns  # noqa: E402

EDGES = [150, 180, 220, 250, 260, 300]
LABELS = ["<150", "150-180", "180-220", "220-250", "250-260", "260-300", ">=300"]
MAIN, NULL = sim.World(1, 1.0, 0.3), sim.World(0, 1.0, 0.3)
WORLDS = {"e03": (MAIN, 101), **{f"e{s}": (MAIN, s) for s in (5101, 5102, 5103, 5104, 5105)},
          "n03": (NULL, 101), **{f"n{s}": (NULL, s) for s in (4101, 4102, 4103, 4104, 4105)},
          # the graded-effect worlds of scripts/estimator_audit.py (beta 0.25, 0.5 on the main seeds; beta 0 on 5101-5105)
          **{f"b025_{s}": (sim.World(1, 0.25, 0.3), s) for s in (101, 5101, 5102, 5103, 5104, 5105)},
          **{f"b050_{s}": (sim.World(1, 0.5, 0.3), s) for s in (101, 5101, 5102, 5103, 5104, 5105)},
          **{f"b000_{s}": (NULL, s) for s in (5101, 5102, 5103, 5104, 5105)}}


def ffill(a: np.ndarray) -> np.ndarray:
    """the latest finite value at or before each hour, per stay"""
    idx = np.where(np.isfinite(a), np.arange(a.shape[1]), -1)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = np.take_along_axis(a, np.maximum(idx, 0), axis=1)
    return np.where(idx >= 0, out, np.nan)


def occupancy(world, seed: int) -> dict:
    p = fns.panel_from_sim(sim.simulate(world, fns.N_TRAIN, seed))
    g, rate = p["glucose_finger"], p["I"]
    r = g[np.isfinite(g)]
    last = ffill(g)
    on = (rate > 0) & np.isfinite(last)
    gd, rd = last[on], rate[on]
    bins_r, bins_d = np.digitize(r, EDGES), np.digitize(gd, EDGES)
    out = {"readings": int(r.size), "dose_hours": int(gd.size), "mean_reading": float(r.mean()),
           "share_readings": {lab: float(np.mean(bins_r == i)) for i, lab in enumerate(LABELS)},
           "share_doses": {lab: float(np.mean(bins_d == i)) for i, lab in enumerate(LABELS)},
           "mean_rate": {lab: (float(rd[bins_d == i].mean()) if (bins_d == i).any() else None) for i, lab in enumerate(LABELS)}}
    out["top_tier_readings"] = float(np.mean(r >= 250))
    out["top_tier_doses"] = float(np.mean(gd >= 250))
    return out


def main() -> None:
    res = {k: occupancy(w, s) for k, (w, s) in WORLDS.items()}
    for kind in ("e", "n", "b025_", "b050_", "b000_"):
        ks = [k for k in res if k.startswith(kind) and not k.startswith("mean")]
        res[f"mean_{kind}"] = {"top_tier_readings": float(np.mean([res[k]["top_tier_readings"] for k in ks])),
                               "top_tier_doses": float(np.mean([res[k]["top_tier_doses"] for k in ks])),
                               "share_doses": {lab: float(np.mean([res[k]["share_doses"][lab] for k in ks])) for lab in LABELS},
                               "share_readings": {lab: float(np.mean([res[k]["share_readings"][lab] for k in ks])) for lab in LABELS},
                               "mean_rate": {lab: float(np.mean([res[k]["mean_rate"][lab] for k in ks])) for lab in LABELS}}
    (ROOT / "results" / "tier_occupancy.json").write_text(json.dumps(res, indent=1) + "\n")
    for k, v in res.items():
        print(k, {"top_r": round(v["top_tier_readings"], 3), "top_d": round(v["top_tier_doses"], 3),
                  "doses": {a: round(b, 3) for a, b in v["share_doses"].items()}})


if __name__ == "__main__":
    main()
