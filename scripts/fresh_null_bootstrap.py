"""Stay bootstrap of the agents' estimators on the five fresh no-effect worlds, for a pooled no-effect bias with an
interval. Each world's training records are written as the agents read them (scripts/fresh_null_seeds.py, records),
resampled by stay with replacement exactly as scripts/replay_bootstrap.py resamples the original no-effect world, and the
agent's own Python runs unchanged on each copy in a fresh sandbox.

    python3 scripts/fresh_null_bootstrap.py <claude_a|openai_a> <first draw> <last draw> [--seeds 4101 ...] [--workers 6]
draw -1 is the world's own records (it must reproduce results/fresh_null_seeds_agents.json). One line per draw is
appended to results/fresh_null_bootstrap/<run>_<seed>.jsonl. CPU only; synthetic data only.
"""
import argparse
import json
import multiprocessing as mp
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
WORK = ROOT / "work" / "fresh_boot"
OUT = ROOT / "results" / "fresh_null_bootstrap"


def world_data(train_seed: int) -> Path:
    """the fresh world's training and tuning records as the agents read them (written once)"""
    import fresh_null_seeds as fns
    d = WORK / f"world_{train_seed}"
    if (d / "train.csv").is_file():
        return d
    tuning_seed = dict(fns.SEEDS)[train_seed]
    d.mkdir(parents=True, exist_ok=True)
    fns.records(train_seed, fns.N_TRAIN, 1_010_000_000).write_csv(d / "train.csv.part", float_precision=4)
    fns.records(tuning_seed, fns.N_TUNING, 2_020_000_000).write_csv(d / "tuning.csv", float_precision=4)
    shutil.copy2(ROOT / "agent_data" / "n03" / "data" / "README.md", d / "README.md")
    (d / "train.csv.part").rename(d / "train.csv")
    return d


def resampled(run: str, train_seed: int, draw: int) -> Path:
    """as replay_bootstrap.resampled: 30,000 stays drawn with replacement, ids base .. base + 29,999 in draw order"""
    import polars as pl
    src = world_data(train_seed)
    if draw < 0:
        return src
    d = WORK / f"data_{run}_{train_seed}_{draw}"
    if (d / "train.csv").is_file():
        return d
    d.mkdir(parents=True, exist_ok=True)
    tr = pl.read_csv(src / "train.csv", infer_schema_length=0).with_row_index("_row")
    ids = tr["stay_id"].unique(maintain_order=True).to_numpy()
    pick = np.random.default_rng(10_000 + draw).choice(ids, size=len(ids), replace=True)
    base = int(ids.astype(np.int64).min())
    m = pl.DataFrame({"stay_id": pick, "new_id": (base + np.arange(len(pick))).astype(str)})
    out = tr.join(m, on="stay_id", how="inner").sort(["new_id", "_row"])
    out = out.with_columns(pl.col("new_id").alias("stay_id")).drop("new_id", "_row")
    out.write_csv(d / "train.csv.part")
    (d / "train.csv.part").rename(d / "train.csv")
    for f in ("tuning.csv", "README.md"):
        shutil.copy2(src / f, d / f)
    return d


def one(args) -> dict:
    run, train_seed, draw = args
    import replay_bootstrap as rb
    from agent.kernel import Kernel
    data = resampled(run, train_seed, draw)
    work = WORK / f"work_{run}_{train_seed}_{draw}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    calls = [json.loads(x) for x in open(ROOT / "agent_runs" / run / "tool_calls.jsonl")]
    cells = [c for c in calls if c.get("type") == "python" and c.get("run") and c["run"] <= rb.LAST[run]]
    k = Kernel(work, ROOT / "agent_data" / "n03" / "model", data, "white")
    outputs, restarts = {}, []
    try:
        for c in cells:
            out, restarted = k.run(c["code"])
            outputs[c["run"]] = out
            if restarted:
                restarts.append(c["run"])
    finally:
        k.kill()
    try:
        v, err = rb.implied(run, outputs), None
    except Exception as e:  # noqa: BLE001
        v, err = None, f"{type(e).__name__}: {e}"
    shutil.rmtree(work, ignore_errors=True)
    if draw >= 0:
        shutil.rmtree(data, ignore_errors=True)
    return {"run": run, "seed": train_seed, "draw": draw, "implied_insulin_x2": v, "restarts": restarts, "error": err}


if __name__ == "__main__":
    import fresh_null_seeds as fns
    ap = argparse.ArgumentParser()
    ap.add_argument("run", choices=("claude_a", "openai_a"))
    ap.add_argument("first", type=int)
    ap.add_argument("last", type=int)
    ap.add_argument("--seeds", type=int, nargs="*", default=[s for s, _ in fns.SEEDS])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for s in a.seeds:                       # write every world's records before any worker starts
        world_data(s)
    todo = []
    for d in range(a.first, a.last + 1):    # draw-major, so every world gets its first draws early
        for s in a.seeds:
            f = OUT / f"{a.run}_{s}.jsonl"
            done = {json.loads(x)["draw"] for x in open(f)} if f.is_file() else set()
            if d not in done:
                todo.append((a.run, s, d))
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex:
        for r in ex.map(one, todo):
            with open(OUT / f"{r['run']}_{r['seed']}.jsonl", "a") as fh:
                fh.write(json.dumps(r) + "\n")
            print(json.dumps(r), flush=True)
