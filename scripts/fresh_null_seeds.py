"""The information-matched script's estimator, unchanged, on no-effect worlds that no development step saw: fresh simulator seeds, records formed as the simulator code's records are (readings rounded to whole units,
rates written to four decimals), a stay bootstrap of the implied insulin x2 shift per seed.

    python scripts/fresh_null_seeds.py check            # the panel built here equals the one read from the no-effect
                                                        # world's training records (seed 101), so fresh seeds are faithful
    python scripts/fresh_null_seeds.py run [--draws 200] [--workers 8]
        -> results/fresh_null_seeds.json (and every copy's value in results/fresh_null_seeds_draws.json)
    python scripts/fresh_null_seeds.py agents    # Claude's and Codex's estimation code, replayed unchanged on the same
        -> results/fresh_null_seeds_agents.json   seeds' records written as the agents read them (one estimate per seed;
                                                  csv_check first: seed 101 written this way equals their train.csv)
The simulator (agent/backend.py, name "sim") is only called, never changed. CPU.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "scripts"))
import backend  # noqa: E402  (the simulator code)
import scripted_estimate as se  # noqa: E402
from backend import sim  # noqa: E402

#: chosen before any of them was simulated: (training seed, tuning seed)
SEEDS = [(4101, 4201), (4102, 4202), (4103, 4203), (4104, 4204), (4105, 4205)]
NO_EFFECT = sim.World(0, 1.0, 0.3)
N_TRAIN, N_TUNING = 30_000, 400
OBS = {"glucose_finger": "FSG", "glucose_lab": "GLU", "map_arterial": "ABP", "map_cuff": "NBP"}
BAND = 1.42                     # a tenth of the true insulin x2 change on the held-out stays (mg/dL), as in the paper


def panel_from_sim(s: dict) -> dict:
    """the scripted estimator's panel (stay x ICU hour 0..25) from the simulator's output, as the records hold it"""
    n, H = len(s["subject"]), se.H
    out = {"stay": s["subject"].astype(np.int64)}
    for e, k in OBS.items():
        a = np.full((n, H), np.nan)
        a[:, :sim.HOURS] = np.round(s[k][:, sim.PRE:sim.PRE + sim.HOURS])
        out[e] = a
    for k, d in (("I", "INS"), ("N", "NE")):
        a = np.zeros((n, H))
        r = np.round(s[d][:, sim.PRE:sim.PRE + sim.HOURS], 4)
        a[:, :sim.HOURS] = np.where(r > 0, r, 0.0)
        out[k] = a
    return out


def check() -> None:
    mine = panel_from_sim(sim.simulate(NO_EFFECT, N_TRAIN, 101))
    ref = se.panel(ROOT / "agent_data" / "n03" / "data" / "train.csv")
    assert np.array_equal(mine["stay"], ref["stay"])
    for k in ("glucose_finger", "glucose_lab", "map_arterial", "map_cuff", "I", "N"):
        same = np.array_equal(np.nan_to_num(mine[k], nan=-1), np.nan_to_num(ref[k], nan=-1))
        print(k, "identical" if same else f"max diff {np.nanmax(np.abs(mine[k] - ref[k]))}")
        assert same, k
    with contextlib.redirect_stdout(io.StringIO()):
        k_ = se.kernel(se.tsls(mine, se.deviations(mine), "I", "glucose_finger"))
    tu = se.panel(ROOT / "agent_data" / "n03" / "data" / "tuning.csv")
    print("implied insulin x2 on the original tuning stays:", round(se.implied(tu, "I", "glucose_finger", k_, 2.0)[0], 4),
          "(results/scripted/estimate_n03.json: 1.051)")


_TR = _TU = None


def _init(train_seed: int, tuning_seed: int) -> None:
    global _TR, _TU
    _TR = panel_from_sim(sim.simulate(NO_EFFECT, N_TRAIN, train_seed))
    _TU = panel_from_sim(sim.simulate(NO_EFFECT, N_TUNING, tuning_seed))


def _one(draw: int) -> float:
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


def run(draws: int, workers: int) -> None:
    out = {"world": NO_EFFECT.name, "band_mg_dl": BAND, "draws_per_seed": draws, "seeds": {}}
    every = {}                                  # each copy's implied shift, for the pooled interval over worlds
    for train_seed, tuning_seed in SEEDS:
        with ProcessPoolExecutor(workers, initializer=_init, initargs=(train_seed, tuning_seed)) as ex:
            v = list(ex.map(_one, range(-1, draws)))
        point, boot = v[0], np.array(v[1:])
        every[str(train_seed)] = [float(x) for x in v]          # draw -1 (the world's own records) first
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out["seeds"][str(train_seed)] = {
            "tuning_seed": tuning_seed, "implied_insulin_x2": round(point, 3), "interval": [round(lo, 3), round(hi, 3)],
            "point_within_band": bool(abs(point) <= BAND), "share_of_copies_within_band": round(float(np.mean(np.abs(boot) <= BAND)), 3),
            "verdict": ("within the band" if -BAND < lo and hi < BAND else
                        "beyond the band" if lo > BAND or hi < -BAND else "straddles the band edge")}
        print(train_seed, json.dumps(out["seeds"][str(train_seed)]), flush=True)
    pts = [s["implied_insulin_x2"] for s in out["seeds"].values()]
    out["share_of_seeds_point_within_band"] = round(float(np.mean([abs(x) <= BAND for x in pts])), 3)
    out["share_of_seeds_interval_within_band"] = round(float(np.mean([s["verdict"] == "within the band" for s in out["seeds"].values()])), 3)
    (ROOT / "results" / "fresh_null_seeds.json").write_text(json.dumps(out, indent=1) + "\n")
    (ROOT / "results" / "fresh_null_seeds_draws.json").write_text(json.dumps(every) + "\n")


def records(seed: int, n: int, base: int):
    """the stays of one seed as the agents read them, stay ids relabelled to base, base + 1, ... (the ids of the original
    split; one agent's code assumes them)"""
    import polars as pl
    sys.path.insert(0, str(ROOT))
    from agent.prepare import events_from
    from backend import event_stream
    s = sim.simulate(NO_EFFECT, n, seed)
    f = events_from(event_stream(s))
    return f.with_columns((pl.col("stay_id") - seed * 10_000_000 + base).alias("stay_id"))


def csv_check() -> None:
    import polars as pl
    d = ROOT / "work" / "fresh_seeds" / "check_101"
    d.mkdir(parents=True, exist_ok=True)
    records(101, N_TRAIN, 1_010_000_000).write_csv(d / "train.csv", float_precision=4)
    same = (d / "train.csv").read_bytes() == (ROOT / "agent_data" / "n03" / "data" / "train.csv").read_bytes()
    print("seed 101 written as the agents read it equals agent_data/n03/data/train.csv byte for byte:", same)
    assert same
    (d / "train.csv").unlink()


def agents() -> None:
    import shutil
    sys.path.insert(0, str(ROOT))
    import replay_bootstrap as rb
    from agent.kernel import Kernel
    csv_check()
    out = {"world": NO_EFFECT.name, "band_mg_dl": BAND, "seeds": {}}
    for train_seed, tuning_seed in SEEDS:
        d = ROOT / "work" / "fresh_seeds" / str(train_seed)
        d.mkdir(parents=True, exist_ok=True)
        records(train_seed, N_TRAIN, 1_010_000_000).write_csv(d / "train.csv", float_precision=4)
        records(tuning_seed, N_TUNING, 2_020_000_000).write_csv(d / "tuning.csv", float_precision=4)
        shutil.copy2(ROOT / "agent_data" / "n03" / "data" / "README.md", d / "README.md")
        row = {}
        for run in ("claude_a", "openai_a"):
            work = d / f"work_{run}"
            shutil.rmtree(work, ignore_errors=True)
            work.mkdir()
            calls = [json.loads(x) for x in open(ROOT / "agent_runs" / run / "tool_calls.jsonl")]
            cells = [c for c in calls if c.get("type") == "python" and c.get("run") and c["run"] <= rb.LAST[run]]
            k = Kernel(work, ROOT / "agent_data" / "n03" / "model", d, "white")
            outputs = {}
            try:
                for c in cells:
                    outputs[c["run"]] = k.run(c["code"])[0]
            finally:
                k.kill()
            try:
                v = rb.implied(run, outputs)
            except Exception as e:  # noqa: BLE001
                v = None
                print(train_seed, run, "failed:", type(e).__name__, e, flush=True)
            row[run] = None if v is None else {"implied_insulin_x2": round(v, 3), "within_band": bool(abs(v) <= BAND)}
            shutil.rmtree(work, ignore_errors=True)
        out["seeds"][str(train_seed)] = row
        print(train_seed, json.dumps(row), flush=True)
        shutil.rmtree(d, ignore_errors=True)
    for run in ("claude_a", "openai_a"):
        v = [r[run]["within_band"] for r in out["seeds"].values() if r.get(run)]
        out[f"{run}_share_of_seeds_within_band"] = round(float(np.mean(v)), 3) if v else None
    (ROOT / "results" / "fresh_null_seeds_agents.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("check", "run", "agents"))
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    {"check": check, "agents": agents}.get(a.cmd, lambda: run(a.draws, a.workers))()
