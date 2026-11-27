"""Claude's two repeat runs in the main world (claude_d, claude_e): the records-based insulin x2 estimate each sized its edit
to, replayed from the agent's own Python runs and read on a world's tuning stays as Eq. (2) (as audit_eq2 does for claude_a).

Both runs estimate glucose_finger's response per unit insulin rate (u/h) by lag from the training stays and apply it to the
tuning stays' doubled doses. Eq. (2) is scripted_estimate.implied: the mean, over tuning stay-hours 13-24 with a finger-stick
reading and insulin run since hour 12, of sum_l psi_l * u_{t-l} [t-l >= 12].

  claude_d  Python run 22 (G-estimation: the dose's deviation from the step policy as instrument, IPW for finger-stick
            sampling) prints psi_1..psi_12 as lines "k n beta b se s". Run 23 applies it, aligned, to its 100 sel1 stays:
            -14.512. Later targets use the same profile but slip one hour (runs 56/63/188: lag k applied to the dose at hour
            t-k-1, the undoubled hour-11 dose included): -13.83 (8 lags) and -13.99 (12 lags) on sel1, the numbers its
            insulin channel was calibrated and checked against. A global K=8 (run 48) truncates its aligned function to 8
            lags: -13.16 on sel2, the final answer's "-13.2". Here: Eq. (2) with all 12 lags, aligned.
  claude_e  Python run 23 deconvolves local projections (run 22) into a direct effect d_1..d_10, printed as
            "direct d (finger): [...]"; it applies it (Eq. (2)) to its sel1 stays, -13.325. Run 151 copies the profile to
            two decimals; run 238 pools it over all 400 tuning stays, -13.545, the final answer's "-13.5". Here: Eq. (2)
            with the run-23 profile (three decimals), lags 11-12 zero.
Runs 1..LAST read only /data, /model and their own /work files. The forward tool's first dose_edit result
(/work/fwd_1) is read by claude_d run 18 and claude_e run 19, which fail in a replay, as do claude_e run 20 and the end of
run 23 (after the profile is printed), which use it; the profiles do not depend on it. No randomness enters a profile.
On the main world both replays reproduce the agents' printed values (main-world check): Eq. (2) with the
replayed profiles gives -14.512 (claude_d, its sel1 stays) and -13.325 (claude_e, its sel1 stays); -13.545 with run 151's
two-decimal copy on all 400 tuning stays. implied() on the 400 tuning stays: claude_d -13.884, claude_e -13.557.
One replay takes about 15 s (claude_d) or 5 s (claude_e), the kernel's start included.

    python3 scripts/replay_reps_claude.py <claude_d|claude_e> [data dir] [model dir]   (default: the main world's)
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
LAST = {"claude_d": 22, "claude_e": 23}
WORK = ROOT / "work" / "replay_reps"


def profile(run: str, outputs: dict[int, str]) -> np.ndarray:
    """psi_1..psi_12 (mg/dL of glucose_finger per u/h of insulin at lag l) as the replayed run printed it"""
    if run == "claude_d":
        b = {int(k): float(v) for k, v in re.findall(r"^(\d+) \d+ beta (\S+) se", outputs[22], flags=re.M)}
        psi = np.array([b[k] for k in range(1, 13)])
    else:
        psi = np.array([float(v) for v in re.search(r"direct d \(finger\):\s*\[([^\]]*)\]", outputs[23]).group(1).split()])
    return np.r_[psi, np.zeros(12 - len(psi))]


def implied(run: str, outputs: dict[int, str], data_dir: Path) -> float:
    """the insulin x2 shift of glucose_finger (mg/dL) the replayed estimate implies on data_dir's tuning stays (Eq. (2))"""
    import audit_eq2
    import scripted_estimate as se
    tp = audit_eq2.tuning_panel(Path(data_dir) / "tuning.csv")
    return se.implied(tp, "I", "glucose_finger", profile(run, outputs), 2.0)[0]


def replay(run: str, data_dir: Path, model_dir: Path, work: Path) -> dict[int, str]:
    """Python runs 1..LAST[run], verbatim, in a fresh sandboxed kernel (as audit_eq2.one); work is emptied first"""
    from agent.kernel import Kernel
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    calls = [json.loads(x) for x in open(ROOT / "agent_runs" / run / "tool_calls.jsonl")]
    k = Kernel(work, model_dir, data_dir, "white")
    outputs = {}
    try:
        for c in calls:
            if c.get("type") == "python" and c.get("run") and c["run"] <= LAST[run]:
                outputs[c["run"]] = k.run(c["code"])[0]
    finally:
        k.kill()
    return outputs


if __name__ == "__main__":
    run = sys.argv[1]
    data = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "agent_data" / "e03" / "data"
    model = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "agent_data" / "e03" / "model"
    t0 = time.time()
    out = replay(run, data, model, WORK / f"work_{run}")
    took = time.time() - t0
    shutil.rmtree(WORK / f"work_{run}", ignore_errors=True)
    (WORK / f"{run}_outputs.json").write_text(json.dumps(out) + "\n")
    print(json.dumps({"run": run, "data": str(data), "psi": profile(run, out).round(3).tolist(),
                      "implied": round(implied(run, out, data), 4), "replay_s": round(took, 1)}), flush=True)
