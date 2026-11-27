"""The estimator audit on the readout's own scale (Eq. 2 of the paper), in no-effect and in main worlds.

Claude's replayed estimation code prints a finger-stick lag profile psi_1..psi_12 (Python run 16); its implied insulin x2
shift is Eq. (2) applied to that profile on the world's tuning stays: the mean over the scored stay-hours (hours 13-24,
finger-stick read, insulin run since hour 12) of sum_l psi_l * u_{t-l} [t-l >= 12] (scripted_estimate.implied). Codex's
code computes Eq. (2) itself, with psi_l = beta * rho^(l-1), on the training stays (its run 24), so its value is read as
printed. The full script's estimator applies Eq. (2) to its kernel on the tuning stays.

Worlds: the original no-effect world (n03) and five fresh ones (seeds 4101-4105, scripts/fresh_null_seeds.py); the original
main world (e03) and five fresh main worlds simulated here (training seeds 5101-5105, tuning seeds 5201-5205, the main
world's simulator settings). In a main world each estimator is compared with the simulator's true insulin x2 shift on the
same stays and stay-hours (tuning stays for Claude and the script, training stays for Codex), every random number fixed.

    python3 scripts/audit_eq2.py check                 # e03 records formed here equal agent_data/e03/data/train.csv
    python3 scripts/audit_eq2.py truth                 # -> results/audit_eq2/truth.json
    python3 scripts/audit_eq2.py replay <claude_a|openai_a> <world,...> <first> <last> [--workers 8]
        -> results/audit_eq2/<run>_<world>.jsonl (draw -1 = the world's own records; copies resample training stays)
    python3 scripts/audit_eq2.py script <world,...> <draws> [--workers 8]   -> results/audit_eq2/script_<world>.json
CPU only; synthetic data only; the simulator is called and never changed.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import io  # noqa: E402
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
import scripted_estimate as se  # noqa: E402
from backend import sim  # noqa: E402

OUT = ROOT / "results" / "audit_eq2"
WORK = ROOT / "work" / "audit_eq2"
MAIN = sim.World(1, 1.0, 0.3)
NULL = sim.World(0, 1.0, 0.3)
N_TRAIN, N_TUNING = 30_000, 400
FRESH_MAIN = [(5101, 5201), (5102, 5202), (5103, 5203), (5104, 5204), (5105, 5205)]
FRESH_NULL = [(4101, 4201), (4102, 4202), (4103, 4203), (4104, 4204), (4105, 4205)]


def worlds() -> dict:
    """name -> (simulator world, training seed, tuning seed, model folder)"""
    w = {"n03": (NULL, 101, 202, "n03"), "e03": (MAIN, 101, 202, "e03")}
    w.update({f"n{a}": (NULL, a, b, "n03") for a, b in FRESH_NULL})
    w.update({f"e{a}": (MAIN, a, b, "e03") for a, b in FRESH_MAIN})
    return w


def records(world, seed: int, n: int, base: int):
    """the stays of one seed as the agents read them (as fresh_null_seeds.records, for any world)"""
    import polars as pl
    from agent.prepare import events_from
    from backend import event_stream
    s = sim.simulate(world, n, seed)
    f = events_from(event_stream(s))
    return f.with_columns((pl.col("stay_id") - seed * 10_000_000 + base).alias("stay_id"))


def data_dir(name: str) -> Path:
    if name in ("n03", "e03"):
        return ROOT / "agent_data" / name / "data"
    if name.startswith("n4"):
        import fresh_null_bootstrap as fnb
        return fnb.world_data(int(name[1:]))
    world, a, b, _ = worlds()[name]
    d = WORK / f"world_{name}"
    if not (d / "train.csv").is_file():
        d.mkdir(parents=True, exist_ok=True)
        records(world, a, N_TRAIN, 1_010_000_000).write_csv(d / "train.csv.part", float_precision=4)
        records(world, b, N_TUNING, 2_020_000_000).write_csv(d / "tuning.csv", float_precision=4)
        shutil.copy2(ROOT / "agent_data" / "e03" / "data" / "README.md", d / "README.md")
        (d / "train.csv.part").rename(d / "train.csv")
    return d


def check() -> None:
    d = WORK / "check_e03"
    d.mkdir(parents=True, exist_ok=True)
    records(MAIN, 101, N_TRAIN, 1_010_000_000).write_csv(d / "train.csv", float_precision=4)
    same = (d / "train.csv").read_bytes() == (ROOT / "agent_data" / "e03" / "data" / "train.csv").read_bytes()
    print("e03 seed 101 formed here equals agent_data/e03/data/train.csv byte for byte:", same, flush=True)
    (d / "train.csv").unlink()
    assert same


def scored_truth(world, seed: int, n: int) -> float:
    """the simulator's true insulin x2 shift of glucose from hour 12, every random number fixed, averaged over the stay-hours
    Eq. (2) averages over (hours 13-24 with a finger-stick, insulin run since hour 12)"""
    import fresh_null_seeds as fns
    s = sim.simulate(world, n, seed)
    p = fns.panel_from_sim(s)
    r = sim.replay(world, n, seed, s, {"INS": 2.0}, 12)
    d = r["edited"]["G"][:, sim.PRE:sim.PRE + sim.HOURS] - r["factual"]["G"][:, sim.PRE:sim.PRE + sim.HOURS]
    R, y, vals = p["I"], p["glucose_finger"], []
    for t in range(13, 25):
        ok = np.isfinite(y[:, t]) & (R[:, 12:t] > 0).any(axis=1)
        vals.append(d[ok, t])
    return float(np.concatenate(vals).mean())


def truth() -> None:
    out = {}
    for name, (world, a, b, _) in worlds().items():
        if world is MAIN:
            out[name] = {"tuning": round(scored_truth(world, b, N_TUNING), 4), "train": round(scored_truth(world, a, N_TRAIN), 4)}
            print(name, out[name], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "truth.json").write_text(json.dumps(out, indent=1) + "\n")


def resampled(name: str, draw: int) -> Path:
    """30,000 training stays drawn with replacement, ids base .. base + 29,999 (as replay_bootstrap.resampled)"""
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


def tuning_panel(csv: Path) -> dict:
    """scripted_estimate.panel with typed columns: polars can infer a rate column as text when its first rows are empty"""
    import polars as pl
    f = pl.read_csv(csv, columns=["stay_id", "hour", "event", "value", "rate"],
                    schema_overrides={"stay_id": pl.Int64, "hour": pl.Float64, "value": pl.Float64, "rate": pl.Float64})
    f = f.filter(pl.col("hour") >= 0).with_columns(pl.col("hour").floor().cast(pl.Int64).alias("h")).filter(pl.col("h") < se.H)
    stays = np.sort(f["stay_id"].unique().to_numpy())
    idx = {s: i for i, s in enumerate(stays)}
    out = {"stay": stays}
    for e in se.EVENTS:
        g = f.filter(pl.col("event") == e)
        a = np.full((len(stays), se.H), np.nan)
        a[[idx[s] for s in g["stay_id"].to_list()], g["h"].to_numpy()] = g["value"].to_numpy()
        out[e] = a
    for e, k in (("insulin", "I"), ("norepinephrine", "N")):
        g = f.filter((pl.col("event") == e) & (pl.col("rate") > 0))
        a = np.zeros((len(stays), se.H))
        np.add.at(a, ([idx[s] for s in g["stay_id"].to_list()], g["h"].to_numpy()), g["rate"].to_numpy())
        out[k] = a
    return out


def one(args) -> dict:
    run, name, draw = args
    import replay_bootstrap as rb
    from agent.kernel import Kernel
    data = resampled(name, draw)
    work = WORK / f"work_{run}_{name}_{draw}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    calls = [json.loads(x) for x in open(ROOT / "agent_runs" / run / "tool_calls.jsonl")]
    cells = [c for c in calls if c.get("type") == "python" and c.get("run") and c["run"] <= rb.LAST[run]]
    k = Kernel(work, ROOT / "agent_data" / worlds()[name][3] / "model", data, "white")
    outputs = {}
    try:
        for c in cells:
            outputs[c["run"]] = k.run(c["code"])[0]
    finally:
        k.kill()
    row = {"run": run, "world": name, "draw": draw}
    try:
        if run == "claude_a":
            psi = np.array([float(v) for v in re.search(r"psi finger\s*\[([^\]]*)\]", outputs[16]).group(1).split()])
            row["psi"] = psi.tolist()
            tp = tuning_panel(data / "tuning.csv")
            row["eq2"] = se.implied(tp, "I", "glucose_finger", psi, 2.0)[0]
            row["mapped"] = rb.implied(run, outputs)           # the mapping of scripts/replay_bootstrap.py, for comparison
        else:
            row["eq2"] = rb.implied(run, outputs)              # Codex's own Eq. (2), on the training stays
        row["error"] = None
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
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        for r in ex.map(one, todo):
            with open(OUT / f"{r['run']}_{r['world']}.jsonl", "a") as fh:
                fh.write(json.dumps(r) + "\n")
            print(json.dumps({k: v for k, v in r.items() if k != "psi"}), flush=True)


_TR = _TU = None


def _init(world_name: str) -> None:
    import fresh_null_seeds as fns
    global _TR, _TU
    world, a, b, _ = worlds()[world_name]
    _TR = fns.panel_from_sim(sim.simulate(world, N_TRAIN, a))
    _TU = fns.panel_from_sim(sim.simulate(world, N_TUNING, b))


def _script_one(draw: int) -> float:
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
    OUT.mkdir(parents=True, exist_ok=True)
    for n in names:
        with ProcessPoolExecutor(workers, initializer=_init, initargs=(n,)) as ex:
            v = list(ex.map(_script_one, range(-1, draws)))
        (OUT / f"script_{n}.json").write_text(json.dumps({"world": n, "values": [float(x) for x in v]}) + "\n")
        print(n, "script point", round(v[0], 3), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("check", "truth", "replay", "script"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "check":
        check()
    elif a.cmd == "truth":
        truth()
    elif a.cmd == "replay":
        replay(a.args[0], a.args[1].split(","), int(a.args[2]), int(a.args[3]), a.workers)
    else:
        script(a.args[0].split(","), int(a.args[1]), a.workers)
