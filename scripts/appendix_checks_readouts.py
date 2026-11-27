"""Two sets of readouts, keys b1 and b2 of the output.

b1: the in-band runs' own writes (claude_a, openai_d) rescaled by one gain to a given estimate (agent_runs/ss_m*): asked
readouts and value NLL (results/agent_eval.jsonl), total NLL (results/model_parts.jsonl), x1.5, x3, x0 and laboratory
glucose (generalize, design doses), doubled insulin in hours 13-16 and 21-24 (table13 rollouts), and x3 below and in the
top dose bin (as scripts/x3_split.py); the sized gains from results/same_shape/tune_m*.json.
b2: the full script's x3, x0, x1.5 and laboratory glucose on seeds 1 and 2 of the main world and in the five further
worlds (each world's own held-out truth).

    python scripts/appendix_checks_readouts.py   -> results/appendix_checks_readouts.json      (CPU; the simulator code's Python, for the truth)
"""
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "agent")]
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import more_worlds_summary as mws  # noqa: E402
import byhour_submissions as bh  # noqa: E402

gs = mws.gs
from backend import sim  # noqa: E402

GEN = "runs/general/s+1_b1_e0.3"
UQ = ("insulin_x3", "insulin_x0", "insulin_x1.5", "insulin_x2_lab")


def unasked(root, tag):
    fs = sorted((root / GEN).glob(f"{tag}_s0_doses*.parquet"))
    rows = pl.concat([pl.read_parquet(f) for f in fs]).unique(subset=["episode", "hour", "edit", "start"], keep="first")
    return {e: gs.score(rows.filter(pl.col("start") == 12), "e03", e, 12)["ratio"] for e in UQ}


def x3_split(tag):
    _, w = gs.WORLD["e03"]
    s = sim.simulate(w, gs.N, gs.SEED)
    ins = np.asarray(s["INS"])
    row_of = {int(x): i for i, x in enumerate(s["subject"].astype(np.int64))}
    g = pl.concat([pl.read_parquet(f) for f in sorted((ROOT / "work" / GEN).glob(f"{tag}_s0_doses*.parquet"))])
    g = g.filter((pl.col("edit") == "insulin_x3") & pl.col("reached")).join(gs.truth("e03", "insulin_x3", 12),
                                                                              on=["episode", "hour"], how="left")
    top = np.array([bool((3 * ins[row_of[int(e)], sim.PRE + 12: sim.PRE + int(h)] > 5.0).any())
                    for e, h in zip(g["episode"].to_list(), g["hour"].to_list(), strict=True)])
    v, t, subj = g["shift"].to_numpy(), g["true_shift"].to_numpy(), g["subject"].to_numpy()
    out = {}
    for name, msk in (("below_top_bin", ~top), ("top_bin", top)):
        s_, inv = np.unique(subj[msk], return_inverse=True)
        tm, tt = np.bincount(inv, v[msk]), np.bincount(inv, t[msk])
        pick = np.random.default_rng(0).integers(0, len(s_), size=(1000, len(s_)))
        b = tm[pick].sum(1) / tt[pick].sum(1)
        out[name] = [float(tm.sum() / tt.sum()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]
    return out


def main():
    ev = {json.loads(x)["method"]: json.loads(x) for x in open(ROOT / "results" / "agent_eval.jsonl")}
    parts = {json.loads(x)["label"]: json.loads(x) for x in open(ROOT / "results" / "model_parts.jsonl")}
    out = {"b1": {}, "b2": {}}
    for tag in ("ss_mclaude_script", "ss_mclaude_claude", "ss_mcodex_script", "ss_mcodex_claude"):
        tune = json.loads((ROOT / "results" / "same_shape" / f"tune_{tag[3:]}.json").read_text())
        e = ev[f"agent_{tag}"]
        out["b1"][tag] = {
            "gain": tune["final"]["gains"]["I"], "fitted_gain": tune["fitted_gains"]["I"], "budget_scale": tune["final"]["scale"],
            "tuning_rounds": [[r["gains"]["I"], r["readouts"]["insulin_x2"]] for r in tune["rounds"]],
            "asked": {q: e[q]["ratio"] for q in ("insulin_x2", "insulin_half", "ne_x2", "ne_half")},
            "value_nll_change": e["value_nll_change"], "total_nll_change": parts[tag]["total_nll_per_stay"]["change"],
            "unasked": unasked(ROOT / "work", tag), "x3_split": x3_split(tag),
            "early": bh.window(ROOT / "work" / "runs/table13/s+1_b1_e0.3" / f"shifts_agent_{tag}_s0.parquet", 13, 16),
            "late": bh.window(ROOT / "work" / "runs/table13/s+1_b1_e0.3" / f"shifts_agent_{tag}_s0.parquet", 21, 24)}
        print(tag, out["b1"][tag]["gain"], {k: round(v[0], 3) for k, v in out["b1"][tag]["asked"].items()},
              {k: round(v[0], 3) for k, v in out["b1"][tag]["unasked"].items()}, flush=True)
    for s in (1, 2):
        out["b2"][f"main_s{s}"] = unasked(ROOT / "work", f"scriptSseed{s}")
    for m in ("m5101", "m5102", "m5103", "m5104", "m5105"):
        mws.use_world(m)
        out["b2"][m] = unasked(ROOT / "work" / "fresh_main" / m, "instscript")
    for k, v in out["b2"].items():
        print(k, {e: [round(x, 3) for x in r] for e, r in v.items()}, flush=True)
    (ROOT / "results" / "appendix_checks_readouts.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
