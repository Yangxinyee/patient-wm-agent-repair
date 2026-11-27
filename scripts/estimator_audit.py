"""The estimator audit for every main-world weight-access run, in any world, including worlds of graded effect.

scripts/audit_eq2.py replays the estimators of each agent's FIRST main-world run (claude_a runs 1-17, openai_a runs
1-24). This module replays any of the six (claude_a, claude_d, claude_e, openai_a, openai_d, openai_e) the same way: the
agent's Python runs 1..LAST verbatim in a fresh kernel on another world's records, its estimate read off as the implied
insulin x2 shift of Eq. (2) (scripts/replay_reps_claude.py and scripts/replay_reps_codex.py say, for the repeats, which
runs and which printed output). In a main or graded world the value minus the simulator's true shift on the same stays and
stay-hours is the estimator's error; in a no-effect world (truth 0) the value is its bias.

Worlds (simulator world, training seed, tuning seed, model folder for the replay kernel):
  n03, n4101-n4105        no-effect (sign 0), as in audit_eq2
  e03, e5101-e5105        main (effect strength beta 1), as in audit_eq2
  b<k>_<seed>             graded: the main world's simulator with insulin and norepinephrine effects scaled by beta,
                          k in {025, 050} (beta 0.25, 0.5), training seed in {101, 5101..5105} with the main world's
                          tuning seed; these share patients and noise with e03/e51xx, so the four strengths are paired
  b000_<seed>             beta 0 on the same seeds (paired no-effect worlds; the original no-effect worlds use 4101-4105)
The simulator is called with its own World(sign=1, beta=..., eps=0.3) and never changed.

    python scripts/estimator_audit.py truth <world,...>                       -> results/estimator_audit/truth.json
    python scripts/estimator_audit.py replay <run> <world,...> <first> <last> [--workers 16]
        -> results/estimator_audit/<run>_<world>.jsonl   (draw -1 = the world's own records; draws resample training stays)
    python scripts/estimator_audit.py script <world,...> <draws> [--workers 16]  -> results/estimator_audit/script_<world>.json
CPU only, no agent call, no API spend, no GPU. Synthetic data only.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "scripts"))
import audit_eq2 as ae  # noqa: E402
import scripted_estimate as se  # noqa: E402
from backend import sim  # noqa: E402

OUT = ROOT / "results" / "estimator_audit"
WORK = ROOT / "work" / "estimator_audit"
MAIN_SEEDS = [(101, 202), (5101, 5201), (5102, 5202), (5103, 5203), (5104, 5204), (5105, 5205)]
GRADED = {"000": 0.0, "025": 0.25, "050": 0.5}
NEW_NULL_SEEDS = [(6101 + i, 6201 + i) for i in range(10)]


def worlds() -> dict:
    w = dict(ae.worlds())
    for k, beta in GRADED.items():
        for a, b in MAIN_SEEDS:
            world = sim.World(0, 1.0, 0.3) if beta == 0 else sim.World(1, beta, 0.3)
            w[f"b{k}_{a}"] = (world, a, b, "n03" if beta == 0 else "e03")
    for a, b in MAIN_SEEDS:            # the confounded world: an unrecorded feed drives glucose and the dose
        w[f"c100_{a}"] = (sim.World(1, 1.0, 0.3, conf=1.0), a, b, "e03")
    for a, b in NEW_NULL_SEEDS:        # ten more no-effect worlds on fresh seeds
        w[f"b000_{a}"] = (sim.World(0, 1.0, 0.3), a, b, "n03")
    return w


def data_dir(name: str) -> Path:
    if name in ae.worlds():
        return ae.data_dir(name)
    world, a, b, _ = worlds()[name]
    d = WORK / f"world_{name}"
    if not (d / "train.csv").is_file():
        d.mkdir(parents=True, exist_ok=True)
        ae.records(world, a, ae.N_TRAIN, 1_010_000_000).write_csv(d / "train.csv.part", float_precision=4)
        ae.records(world, b, ae.N_TUNING, 2_020_000_000).write_csv(d / "tuning.csv", float_precision=4)
        shutil.copy2(ROOT / "agent_data" / "e03" / "data" / "README.md", d / "README.md")
        (d / "train.csv.part").rename(d / "train.csv")
    return d


def resampled(name: str, draw: int) -> Path:
    """as audit_eq2.resampled, for any world here"""
    import polars as pl
    src = data_dir(name)
    if draw < 0:
        return src
    d = WORK / f"data_{name}_{draw}"
    if (d / "train.csv").is_file():
        return d
    d.mkdir(parents=True, exist_ok=True)
    tr = pl.read_csv(src / "train.csv", infer_schema_length=0).with_row_index("_row")
    ids = tr["stay_id"].unique(maintain_order=True).to_numpy()
    pick = np.random.default_rng(10_000 + draw).choice(ids, size=len(ids), replace=True)
    base = int(ids.astype(np.int64).min())
    m = pl.DataFrame({"stay_id": pick, "new_id": (base + np.arange(len(pick))).astype(str)})
    out = tr.join(m, on="stay_id", how="inner").sort(["new_id", "_row"])
    out.with_columns(pl.col("new_id").alias("stay_id")).drop("new_id", "_row").write_csv(d / "train.csv.part")
    (d / "train.csv.part").rename(d / "train.csv")
    for f in ("tuning.csv", "README.md"):
        shutil.copy2(src / f, d / f)
    return d


def registry() -> dict:
    """run -> (last Python run to replay, implied(outputs, data_dir) -> mg/dL, stays the estimate averages over)"""
    import replay_bootstrap as rb

    def claude_a(outputs, data):
        psi = np.array([float(v) for v in re.search(r"psi finger\s*\[([^\]]*)\]", outputs[16]).group(1).split()])
        return se.implied(ae.tuning_panel(data / "tuning.csv"), "I", "glucose_finger", psi, 2.0)[0]

    reg = {"claude_a": (rb.LAST["claude_a"], claude_a, "tuning"),
           "openai_a": (rb.LAST["openai_a"], lambda o, d: rb.implied("openai_a", o), "train")}
    try:
        import replay_reps_claude as rc
        for r in rc.LAST:
            reg[r] = (rc.LAST[r], (lambda r_: lambda o, d: rc.implied(r_, o, d))(r), getattr(rc, "STAYS", {}).get(r, "tuning"))
    except ImportError:
        pass
    try:
        import replay_reps_codex as rx
        for r in rx.LAST:          # Eq. (2) of the profile on all 400 tuning stays, as for Claude's estimators
            reg[r] = (rx.LAST[r], (lambda r_: lambda o, d: rx.implied(r_, o, d, stays="tuning"))(r), "tuning")
    except ImportError:
        pass
    return reg


def cells(run: str, last: int) -> list[tuple[int, str]]:
    """(Python run, code) for runs 1..last: verbatim, except the Codex repeats' forward-tool reads that only printed a
    file's head (scripts/replay_reps_codex.py PATCH)"""
    try:
        import replay_reps_codex as rx
        if run in rx.LAST:
            return rx.cells(run)
    except ImportError:
        pass
    calls = [json.loads(x) for x in open(ROOT / "agent_runs" / run / "tool_calls.jsonl")]
    return [(c["run"], c["code"]) for c in calls if c.get("type") == "python" and c.get("run") and c["run"] <= last]


def one(args) -> dict:
    run, name, draw = args
    from agent.kernel import Kernel
    last, implied, _ = registry()[run]
    data = resampled(name, draw)
    work = WORK / f"work_{run}_{name}_{draw}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    k = Kernel(work, ROOT / "agent_data" / worlds()[name][3] / "model", data, "white")
    outputs = {}
    try:
        for n, code in cells(run, last):
            outputs[n] = k.run(code)[0]
    finally:
        k.kill()
    row = {"run": run, "world": name, "draw": draw}
    try:
        row["eq2"], row["error"] = float(implied(outputs, data)), None
    except Exception as e:  # noqa: BLE001
        row["eq2"], row["error"] = None, f"{type(e).__name__}: {e}"
    shutil.rmtree(work, ignore_errors=True)
    if draw >= 0:
        shutil.rmtree(data, ignore_errors=True)
    return row


def replay(run: str, names: list[str], first: int, last: int, workers: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for n in names:                                   # every world's records before any worker starts
        data_dir(n)
    todo = []
    for d in range(first, last + 1):
        for n in names:
            f = OUT / f"{run}_{n}.jsonl"
            done = {json.loads(x)["draw"] for x in open(f)} if f.is_file() else set()
            if d not in done:
                todo.append((run, n, d))
    from concurrent.futures import as_completed
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(one, t): t for t in todo}
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001  (e.g. a kernel that did not start under load: left for a rerun)
                print(json.dumps({"task": list(futs[f]), "skipped": f"{type(e).__name__}: {e}"}), flush=True)
                continue
            with open(OUT / f"{r['run']}_{r['world']}.jsonl", "a") as fh:
                fh.write(json.dumps(r) + "\n")
            print(json.dumps(r), flush=True)


_TR = _TU = None


def _init(name: str) -> None:
    import fresh_null_seeds as fns
    global _TR, _TU
    world, a, b, _ = worlds()[name]
    _TR = fns.panel_from_sim(sim.simulate(world, ae.N_TRAIN, a))
    _TU = fns.panel_from_sim(sim.simulate(world, ae.N_TUNING, b))


def _script_one(draw: int) -> float:
    """the full script's estimator (audit_eq2._script_one) on this module's worlds"""
    import contextlib
    import io
    n = len(_TR["stay"])
    if draw < 0:
        p = _TR
    else:
        idx = np.random.default_rng(10_000 + draw).integers(0, n, n)
        p = {k: v[idx] for k, v in _TR.items()}
        p["stay"] = np.arange(n)
    with contextlib.redirect_stdout(io.StringIO()):
        k = se.kernel(se.tsls(p, se.deviations(p), "I", "glucose_finger"))
    return se.implied(_TU, "I", "glucose_finger", k, 2.0)[0]


def script(names: list[str], draws: int, workers: int) -> None:
    """-> results/estimator_audit/script_<world>.json, values[0] the world's own records, then draws 0..draws-1"""
    OUT.mkdir(parents=True, exist_ok=True)
    for n in names:
        with ProcessPoolExecutor(workers, initializer=_init, initargs=(n,), mp_context=mp.get_context("spawn")) as ex:
            v = list(ex.map(_script_one, range(-1, draws)))
        (OUT / f"script_{n}.json").write_text(json.dumps({"world": n, "values": [float(x) for x in v]}) + "\n")
        print(n, "script point", round(v[0], 3), flush=True)


def truth(names: list[str]) -> None:
    """the simulator's true insulin x2 shift on the tuning and training stays' scored stay-hours (audit_eq2.scored_truth)"""
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "truth.json"
    out = json.loads(f.read_text()) if f.is_file() else {}
    for n in names:
        world, a, b, _ = worlds()[n]
        out[n] = {"tuning": ae.scored_truth(world, b, ae.N_TUNING), "train": ae.scored_truth(world, a, ae.N_TRAIN)}
        print(n, out[n], flush=True)
    f.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("truth", "replay", "script"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    if a.cmd == "truth":
        truth(a.args[0].split(","))
    elif a.cmd == "script":
        script(a.args[0].split(","), int(a.args[1]), a.workers)
    else:
        replay(a.args[0], a.args[1].split(","), int(a.args[2]), int(a.args[3]), a.workers)
