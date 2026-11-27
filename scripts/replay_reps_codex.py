"""Codex's two repeat runs in the main world (openai_d, openai_e): the records-based insulin x2 estimate each sized its edit to,
replayed from its own Python runs and turned into Eq. (2) of the paper (as audit_eq2 does for claude_a and openai_a).

  openai_d  run 5: a controlled impulse response of glucose_finger per unit insulin rate at lags 1..12 h (cross-fitted local
            projections on run 4's nuisance fits, the feedback of later doses subtracted), printed as the "controlled_impulse"
            column of its insulin table (4 decimals). Run 6 applied it to the recorded doses (Eq. 2): -13.599 on the 554 scored
            stay-hours of tuning stays 2020000000..099 (the target it compared its edits to), -13.965 on all 400 tuning stays,
            -14.178 on the training stays.
  openai_e  run 5: a fuzzy discontinuity at glucose 180/220 pooled over horizons 1..6 h, fitted as beta and persistence rho
            ("RD dynamic fit [beta, persistence]: [-2.2778 0.85743]"). Run 10 applied psi_l = beta * rho^(l-1) to the recorded
            doses of tuning stays 2020000000..099 at the scored stay-hours: -13.586 (554 stay-hours), its working target
            ("insulin about -13.6/+6.8").
Scored stay-hours (both agents, and scripted_estimate.implied): hours 13-24 with a finger-stick reading and insulin run at an hour
in [12, t). With the agents' full-precision profiles, scripted_estimate.implied gives their printed values exactly; from the printed
profile they move by < 1e-4.

Each run 5 first reads a forward-tool file (/work/fwd_*), only to print its head, and a verbatim replay stops there; PATCH removes
those lines and nothing else. Runs 1-4 read only the records, the model files and what earlier runs made. Runs 6 and 10 read
forward-tool files (saved states, the rollout's rows) and are not replayed: Eq. (2) is computed here instead.

    python scripts/replay_reps_codex.py [openai_d,openai_e] [world]   # replays on a world's records (default e03), checks, times
CPU only; synthetic data only.
"""
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "agent"), str(ROOT / "scripts")]
LAST = {"openai_d": 5, "openai_e": 5}
PATCH = {   # (run, Python run) -> the forward-tool read removed before the replay
    ("openai_d", 5): '''print("=== FORWARD CSV SCHEMAS ===")
for path in ["/work/fwd_1/factual_original.csv", "/work/fwd_2/dose_edit_original.csv"]:
    sample = pd.read_csv(path)
    print(path, sample.shape, sample.columns.tolist())
    print(sample.head(3).to_string(index=False))
''',
    ("openai_e", 5): '''baseline_rollout = pd.read_csv("/work/fwd_1/dose_edit_original.csv")
print("=== ROLLOUT FILE ===")
print(baseline_rollout.head(3).to_string(index=False))
''',
}
NUM = r"[-+0-9.eE]+"


def cells(run: str) -> list[tuple[int, str]]:
    """(Python run, code) for runs 1..LAST, as the agent ran them except for PATCH"""
    calls = [json.loads(x) for x in open(ROOT / "agent_runs" / run / "tool_calls.jsonl")]
    out = []
    for c in calls:
        if c.get("type") == "python" and c.get("run") and c["run"] <= LAST[run]:
            code, cut = c["code"], PATCH.get((run, c["run"]))
            if cut:
                assert code.count(cut) == 1, (run, c["run"])
                code = code.replace(cut, "")
            out.append((c["run"], code))
    return out


def psi(run: str, outputs: dict) -> np.ndarray:
    """the agent's glucose_finger response per unit insulin rate at lags 1..12 h, as printed by its run 5"""
    text = outputs[5]
    if run == "openai_d":
        block = text.split("\n insulin\n", 1)[1].split("Geometric impulse fit", 1)[0]
        rows = re.findall(rf"^\s*(\d+)\s+{NUM}\s+({NUM})\s+{NUM}\s+{NUM}\s+\d+\s*$", block, re.M)
        assert [int(r[0]) for r in rows] == list(range(1, 13)), rows
        return np.array([float(r[1]) for r in rows])
    beta, rho = (float(v) for v in re.search(rf"RD dynamic fit \[beta, persistence\]: \[\s*({NUM})\s+({NUM})\s*\]", text).groups())
    return beta * rho ** np.arange(12)


def implied(run: str, outputs: dict, data_dir: Path, stays: str = "tuning100") -> float:
    """Eq. (2): the insulin x2 shift (mg/dL) the agent's profile implies, the mean over the scored stay-hours of
    sum_l psi_l * u_{t-l}. stays: "tuning100" (tuning stays 2020000000..099, where both agents set their target),
    "tuning" (all 400 tuning stays) or "train" (the training stays)."""
    import audit_eq2
    import scripted_estimate as se
    p = audit_eq2.tuning_panel(Path(data_dir) / ("train.csv" if stays == "train" else "tuning.csv"))
    if stays == "tuning100":
        p = {k: v[:100] for k, v in p.items()}                 # stay ids sorted; the first 100 are 2020000000..099
    return se.implied(p, "I", "glucose_finger", psi(run, outputs), 2.0)[0]


def replay(run: str, data_dir: Path, model_dir: Path, work: Path) -> dict:
    """{Python run: output} of runs 1..LAST in a fresh kernel, as audit_eq2.one"""
    from agent.kernel import Kernel
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    k = Kernel(work, model_dir, data_dir, "white")
    try:
        return {r: k.run(code)[0] for r, code in cells(run)}
    finally:
        k.kill()


def _printed(run: str) -> dict:
    """the agent's own Eq. (2) values as it printed them at the time (runs 6 and 10)"""
    calls = [json.loads(x) for x in open(ROOT / "agent_runs" / run / "tool_calls.jsonl")]
    out = {c["run"]: c["output"] for c in calls if c.get("type") == "python"}
    if run == "openai_d":
        g = {k: re.search(rf"{pat} {{'x2': ({NUM})", out[6]).group(1) for k, pat in
             (("tuning100", "Exact forward readout rows insulin"), ("tuning", "Record-based target insulin tuning_all"),
              ("train", "Record-based target insulin train"))}
    else:
        g = {"tuning100": re.search(rf"insulin 0\.857\d* {{[^}}]*'recorded_hours': ({NUM})", out[10]).group(1)}
    return {"printed": {k: float(v) for k, v in g.items()}, "outputs": out}


if __name__ == "__main__":
    runs = sys.argv[1].split(",") if len(sys.argv) > 1 else sorted(LAST)
    world = sys.argv[2] if len(sys.argv) > 2 else "e03"
    data, model = ROOT / "agent_data" / world / "data", ROOT / "agent_data" / "e03" / "model"
    for run in runs:
        t0 = time.time()
        outs = replay(run, data, model, ROOT / "work" / "replay_reps" / f"{run}_{world}")
        wall = time.time() - t0
        row = {"run": run, "world": world, "wall_s": round(wall, 1),
               "implied": {s: implied(run, outs, data, s) for s in ("tuning100", "tuning", "train")}}
        if world == "e03":
            ag = _printed(run)
            row["agent_printed"] = ag["printed"]
            row["same_profile_as_agent_run5"] = bool(np.array_equal(psi(run, outs), psi(run, ag["outputs"])))
        print(json.dumps(row), flush=True)
