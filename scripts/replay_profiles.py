"""Each main-world agent estimator (used only if the installation-residual rule holds), replayed verbatim on the five further main worlds' training records (the estimator audit's
worlds e5101-e5105), its printed lag profile kept, and its implied insulin x2 shift on that world's held-out stays (Eq. 2,
held-out seed 530x) over the simulator's true shift on the same kind of stay-hours: the readout an installation of it
would give, within the installation residual measured by the installs in the further main worlds.

    python scripts/replay_profiles.py     (CPU, 3 workers)  -> results/replay_profiles.json
"""
import json
import multiprocessing as mp
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "agent"), str(ROOT / "scripts")]
RUNS = ("claude_a", "claude_d", "claude_e", "openai_a", "openai_d", "openai_e")
WORLDS = [f"e{5100 + i}" for i in range(1, 6)]
OUT = ROOT / "results" / "replay_profiles.json"
WORK = ROOT / "work" / "replay_profiles"


def profile(run: str, o: dict) -> list[float]:
    if run == "claude_a":
        psi = [float(v) for v in re.search(r"psi finger\s*\[([^\]]*)\]", o[16]).group(1).split()]
    elif run == "openai_a":
        b, r = (float(x) for x in re.search(r"RD dynamic fit \[effect, persistence\]: \[\s*([-0-9.e]+)\s+([-0-9.e]+)\s*\]", o[23]).groups())
        psi = list(b * r ** np.arange(12))
    elif run in ("claude_d", "claude_e"):
        import replay_reps_claude as rc
        psi = list(rc.profile(run, o))
    else:
        import replay_reps_codex as rx
        psi = list(rx.psi(run, o))
    return [float(x) for x in (psi + [0.0] * 12)[:12]]


def one(args) -> dict:
    run, world = args
    import estimator_audit as ea
    from agent.kernel import Kernel
    last = ea.registry()[run][0]
    data = ea.data_dir(world)
    work = WORK / f"{run}_{world}_{time.strftime('%H%M%S')}"          # a new folder; nothing is deleted
    work.mkdir(parents=True)
    k = Kernel(work, ROOT / "agent_data" / "e03" / "model", data, "white")
    outs = {}
    try:
        for n, code in ea.cells(run, last):
            outs[n] = k.run(code)[0]
    finally:
        k.kill()
    return {"run": run, "world": world, "psi": profile(run, outs)}


def main() -> None:
    import audit_eq2 as ae
    import fresh_null_seeds as fns
    import scripted_estimate as se
    from backend import sim
    res = json.loads(OUT.read_text()) if OUT.is_file() else {"rows": []}
    done = {(r["run"], r["world"]) for r in res["rows"]}
    todo = [(r, w) for r in RUNS for w in WORLDS if (r, w) not in done]
    with ProcessPoolExecutor(3, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(one, t): t for t in todo}
        for f in as_completed(futs):
            try:
                row = f.result()
            except Exception as e:  # noqa: BLE001
                print(json.dumps({"task": list(futs[f]), "skipped": f"{type(e).__name__}: {e}"}), flush=True)
                continue
            ho = 5300 + int(row["world"][1:]) - 5100
            p = fns.panel_from_sim(sim.simulate(ae.MAIN, 600, ho))
            row["implied_held_out"] = float(se.implied(p, "I", "glucose_finger", np.array(row["psi"]), 2.0)[0])
            row["truth_held_out"] = ae.scored_truth(ae.MAIN, ho, 600)
            row["ratio_held_out"] = row["implied_held_out"] / row["truth_held_out"]
            res["rows"].append(row)
            OUT.write_text(json.dumps(res, indent=1) + "\n")
            print(row["run"], row["world"], round(row["ratio_held_out"], 3), flush=True)


if __name__ == "__main__":
    main()
