"""How much of each main-world edit's error does its estimator explain?

For each of the six main-world weight-access runs, on the original main world (e03):
  estimate      the implied insulin x2 shift of the agent's own estimation code, replayed unchanged (Eq. 2), on the 400
                tuning stays, as a fraction of the simulator's true shift on the same stay-hours (results/audit_eq2/truth.json);
                Codex's first run prints its estimate on the training stays, so its profile psi_l = beta * rho^(l-1) is
                re-applied here to the tuning stays, like every other run's
  edit          the edited model's held-out readout, insulin x2, as a fraction of the truth (results/agent_eval.jsonl)
  installation  edit / estimate: the share of its own estimate the installed edit delivers
The estimate and the edit are read on different stays (tuning, held-out), each as a fraction of its own truth.

    python scripts/decompose_edits.py   -> results/decompose_edits.json     (CPU, seconds; the simulator code's Python)
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
sys.path.insert(0, str(ROOT / "agent")); sys.path.insert(0, str(ROOT / "scripts"))
import audit_eq2 as ae  # noqa: E402
import scripted_estimate as se  # noqa: E402

RUNS = ("claude_a", "claude_d", "claude_e", "openai_a", "openai_d", "openai_e")


def implied_tuning(run: str) -> float:
    if run == "openai_a":
        o23 = next(json.loads(x)["output"] for x in open(ROOT / "agent_runs" / "replay" / "openai_a_e03" / "outputs.jsonl")
                   if json.loads(x)["run"] == 23)
        b, r = (float(x) for x in re.search(r"RD dynamic fit \[effect, persistence\]: \[\s*([-0-9.e]+)\s+([-0-9.e]+)\]", o23).groups())
        psi = b * r ** np.arange(12)
        return float(se.implied(ae.tuning_panel(ROOT / "agent_data" / "e03" / "data" / "tuning.csv"), "I", "glucose_finger", psi, 2.0)[0])
    f = (R / "audit_eq2" / "claude_a_e03.jsonl") if run == "claude_a" else (R / "estimator_audit" / f"{run}_e03.jsonl")
    return float(next(json.loads(x)["eq2"] for x in open(f) if json.loads(x)["draw"] == -1))


def main() -> None:
    truth = json.loads((R / "audit_eq2" / "truth.json").read_text())["e03"]["tuning"]
    ev = {json.loads(x)["run"]: json.loads(x) for x in open(R / "agent_eval.jsonl")}
    out = {"truth_tuning": truth, "runs": {}}
    for run in RUNS:
        est = implied_tuning(run)
        e = ev[run]["insulin_x2"]
        edit = e["mean_shift"][0] / e["true_shift"][0]
        out["runs"][run] = {"implied_tuning": est, "estimate_ratio": est / truth, "edit_ratio": edit,
                            "edit_interval": e["ratio"][1:], "installation_ratio": edit / (est / truth)}
        print(run, {k: round(v, 3) if isinstance(v, float) else v for k, v in out["runs"][run].items()}, flush=True)
    est = np.array([v["estimate_ratio"] for v in out["runs"].values()])
    edit = np.array([v["edit_ratio"] for v in out["runs"].values()])
    out["corr_estimate_edit"] = float(np.corrcoef(est, edit)[0, 1])
    (R / "decompose_edits.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
