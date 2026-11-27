"""Norepinephrine in the six main-world runs, estimate against installation on the tuning stays.

For each run: the doubled-norepinephrine shift of arterial MAP its final edit gives on all 400 tuning stays (the agents' own
dose_edit tool, seed 0: results/same_stays/<run>/summary.json), and, where the transcript states one, its own estimate of
that shift on the tuning stays:
  Codex run 2   Eq. (2) for norepinephrine of the controlled impulse response its Python run 5 printed (lags 1-12)
  Codex run 3   the "working NE x2 target" range its notebook states
  Claude run 1  its final answer's "True (from records)" value
The simulator's true shift on the same kind of stay-hours (hours 13-24 with an arterial reading, norepinephrine run since
hour 12; every random number fixed), as audit_eq2.scored_truth does for insulin.

    python scripts/ne_installation.py   -> results/ne_installation.json     (CPU, seconds; synthetic data only)
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "agent"), str(ROOT / "scripts")]
import audit_eq2 as ae  # noqa: E402
import fresh_null_seeds as fns  # noqa: E402
import scripted_estimate as se  # noqa: E402
from backend import sim  # noqa: E402

RUNS = ("claude_a", "claude_d", "claude_e", "openai_a", "openai_d", "openai_e")


def calls(run: str) -> list[dict]:
    return [json.loads(x) for x in open(ROOT / "agent_runs" / run / "tool_calls.jsonl")]


def truth_ne(factor: float) -> float:
    s = sim.simulate(ae.MAIN, ae.N_TUNING, 202)
    p = fns.panel_from_sim(s)
    r = sim.replay(ae.MAIN, ae.N_TUNING, 202, s, {"NE": factor}, 12)
    d = r["edited"]["M"][:, sim.PRE:sim.PRE + sim.HOURS] - r["factual"]["M"][:, sim.PRE:sim.PRE + sim.HOURS]
    R, y, vals = p["N"], p["map_arterial"], []
    for t in range(13, 25):
        ok = np.isfinite(y[:, t]) & (R[:, 12:t] > 0).any(axis=1)
        vals.append(d[ok, t])
    return float(np.concatenate(vals).mean())


def truth_ins(factor: float) -> float:
    """as audit_eq2.scored_truth (insulin x2), for any factor"""
    s = sim.simulate(ae.MAIN, ae.N_TUNING, 202)
    p = fns.panel_from_sim(s)
    r = sim.replay(ae.MAIN, ae.N_TUNING, 202, s, {"INS": factor}, 12)
    d = r["edited"]["G"][:, sim.PRE:sim.PRE + sim.HOURS] - r["factual"]["G"][:, sim.PRE:sim.PRE + sim.HOURS]
    R, y, vals = p["I"], p["glucose_finger"], []
    for t in range(13, 25):
        ok = np.isfinite(y[:, t]) & (R[:, 12:t] > 0).any(axis=1)
        vals.append(d[ok, t])
    return float(np.concatenate(vals).mean())


def main() -> None:
    tun = ae.tuning_panel(ROOT / "agent_data" / "e03" / "data" / "tuning.csv")
    out = {"truth_ne_x2_tuning": truth_ne(2.0), "runs": {},
           "truth_tuning": {"insulin_x2": truth_ins(2.0), "insulin_half": truth_ins(0.5), "ne_x2": truth_ne(2.0), "ne_half": truth_ne(0.5)}}
    for run in RUNS:
        sm = json.loads((ROOT / "results" / "same_stays" / run / "summary.json").read_text())["edited"]["edits"]
        out["runs"][run] = {"ne_x2_tuning_readout": sm["ne_x2"]["mean_shift"], "ne_half_tuning_readout": sm["ne_half"]["mean_shift"]}
    # Codex run 2: the controlled impulse response of norepinephrine, Python run 5
    o5 = next(c["output"] for c in calls("openai_d") if c.get("type") == "python" and c.get("run") == 5)
    block = o5.split("\n norepinephrine\n", 1)[1].split("Geometric impulse fit", 1)[0]
    rows = re.findall(r"^\s*(\d+)\s+[-0-9.]+\s+([-0-9.]+)\s+[-0-9.]+\s+[-0-9.]+\s+\d+\s*$", block, re.M)
    assert [int(r[0]) for r in rows] == list(range(1, 13)), rows
    psi = np.array([float(r[1]) for r in rows])
    out["runs"]["openai_d"]["ne_x2_estimate"] = float(se.implied(tun, "N", "map_arterial", psi, 2.0)[0])
    # Codex run 3: the working target range in its notes
    txt = "\n".join(c.get("output") or "" for c in calls("openai_e") if c.get("type") == "python")
    lo, hi = (float(x) for x in re.search(r"Working NE x2 target about \+([0-9.]+) to \+([0-9.]+)", txt).groups())
    out["runs"]["openai_e"]["ne_x2_estimate_range"] = [lo, hi]
    # Claude run 1: its stated true readout
    fa = (ROOT / "agent_runs" / "claude_a" / "final_answer.md").read_text()
    out["runs"]["claude_a"]["ne_x2_estimate"] = float(re.search(r"Norepinephrine x2 → map_arterial \| [-+0-9.]+ \| [-+−0-9.]+ \| ([-+0-9.]+)", fa).group(1))
    for r, v in out["runs"].items():
        e = v.get("ne_x2_estimate") or (np.mean(v["ne_x2_estimate_range"]) if "ne_x2_estimate_range" in v else None)
        if e:
            v["installed_share"] = v["ne_x2_tuning_readout"] / e
    (ROOT / "results" / "ne_installation.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
