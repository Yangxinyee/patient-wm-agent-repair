"""Every scored agent submission in one record: the six main-world runs (seed 0, main instruction),
the eight told runs (seeds 1 and 2) and the nine scored further-world runs (five further worlds, the instruction alternating);
claude_w3 (paused, no edit) and the rerun claude_w3r are listed apart and never counted.

Per submission: setting, instruction, doubled insulin over the truth with its 95% interval and the point and interval
verdicts against the band (0.9-1.1, half-up to two decimals), x3 and x0, total-NLL change per stay, declined replies,
whether it left insulin as the unedited model of its world and seed had it (largest change of the x2, x3 and x0 readouts
<= 0.01), the full script's doubled-insulin readout on the same trained model, and the paired difference agent minus
script on the same held-out stays (stays resampled jointly, 2,000 draws, models fixed), as in scripts/paired_readout.py.

    python scripts/submissions.py   -> results/submissions.json      (CPU; the simulator code's Python, for the truth)
"""
import json
import os
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "agent")]
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import more_worlds_summary as mws  # noqa: E402   (patches the truth replay for the further worlds' held-out ids)

gs = mws.gs
N_DRAWS, SEED = 2000, 0
MAIN = [("claude_a", "Claude", 0), ("claude_d", "Claude", 0), ("claude_e", "Claude", 0),
        ("openai_a", "Codex", 0), ("openai_d", "Codex", 0), ("openai_e", "Codex", 0)]
GEN_KEY = {"claude_a": "e03:Claude, white box", "claude_d": "e03:Claude, white box, run 2",
           "claude_e": "e03:Claude, white box, run 3", "openai_a": "e03:Codex, white box",
           "openai_d": "e03:Codex, white box, run 2", "openai_e": "e03:Codex, white box, run 3"}
TOLD = ["claude_i", "claude_j", "claude_k", "claude_l", "openai_i", "openai_j", "openai_k", "openai_l"]
T13 = "runs/table13/s+1_b1_e0.3"


def hu(x):
    return Decimal(repr(float(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def verdicts(r):
    p, lo, hi = hu(r[0]), hu(r[1]), hu(r[2])
    iv = ("within" if Decimal("0.9") <= lo and hi <= Decimal("1.1")
          else "beyond" if hi < Decimal("0.9") or lo > Decimal("1.1") else "straddles")
    return ("within" if Decimal("0.9") <= p <= Decimal("1.1") else "outside"), iv


def per_stay(path, world_root):
    g = pl.read_parquet(path).filter((pl.col("edit") == "insulin_x2") & pl.col("reached") & (pl.col("hour") <= 24))
    g = g.join(gs.truth("e03", "insulin_x2", 12), on=["episode", "hour"], how="left")
    return g.group_by("episode").agg(pl.col("shift").sum().alias("m"), pl.col("true_shift").sum().alias("t"),
                                     pl.len().alias("n")).sort("episode")


def paired(a, b):
    j = a.join(b, on="episode", suffix="_b")
    j = j.filter(pl.col("n") == pl.col("n_b"))                    # stay-hours both rollouts reached
    ma, mb, t = (j[c].to_numpy() for c in ("m", "m_b", "t"))
    pick = np.random.default_rng(SEED).integers(0, len(t), size=(N_DRAWS, len(t)))
    T = t[pick].sum(1)
    d = ma[pick].sum(1) / T - mb[pick].sum(1) / T
    lo, hi = np.quantile(d, [0.025, 0.975])
    return {"agent": round(float(ma.sum() / t.sum()), 4), "script": round(float(mb.sum() / t.sum()), 4),
            "difference": [round(float((ma.sum() - mb.sum()) / t.sum()), 4), round(float(lo), 4), round(float(hi), 4)],
            "n_stays": int(len(t))}


def main():
    ev = {json.loads(x)["method"]: json.loads(x) for x in open(ROOT / "results" / "agent_eval.jsonl")}
    parts = {json.loads(x)["label"]: json.loads(x) for x in open(ROOT / "results" / "model_parts.jsonl")}
    readout = [json.loads(x) for x in open(ROOT / "results" / "readout.jsonl")]
    ctl = lambda m, s: next(r for r in readout if r["key"] == "e03" and r["method"] == m and r["seed"] == s)  # noqa: E731
    gen = json.loads((ROOT / "results" / "generalize.json").read_text())
    pa = json.loads((ROOT / "results" / "told_runs_summary.json").read_text())["runs"]
    pss = json.loads((ROOT / "results" / "told_runs_plain_seeds.json").read_text())["seeds"]
    cs = json.loads((ROOT / "results" / "more_worlds_summary.json").read_text())
    asg = {r["run"]: r for r in json.loads((ROOT / "logs" / "more_worlds" / "assignment.json").read_text())["runs"]}
    unedited = {s: {"insulin_x2": ctl("es_register", s)["insulin_x2"]["ratio"][0],
                    "insulin_x3@12": pss[str(s)]["insulin_x3@12"][0], "insulin_x0@12": pss[str(s)]["insulin_x0@12"][0]}
                for s in (0, 1, 2)}
    rows = []

    def add(run, agent, setting, world_root, seed, message, x3, x0, script_x2, script_file):
        e = ev[f"agent_{run}"]
        x2 = e["insulin_x2"]["ratio"]
        st = json.loads((ROOT / "agent_runs" / run / "state.json").read_text())
        un = (unedited[seed] if world_root is None else
              {"insulin_x2": cs["worlds"][world_root]["unedited"]["asked"]["insulin_x2"][0],
               "insulin_x3@12": cs["worlds"][world_root]["unedited"]["unqueried"]["insulin_x3@12"][0],
               "insulin_x0@12": cs["worlds"][world_root]["unedited"]["unqueried"]["insulin_x0@12"][0]})
        dmax = max(abs(x2[0] - un["insulin_x2"]), abs(x3 - un["insulin_x3@12"]), abs(x0 - un["insulin_x0@12"]))
        base = ROOT / "work" if world_root is None else ROOT / "work" / "fresh_main" / world_root
        pr = paired(per_stay(base / T13 / f"shifts_agent_{run}_s0.parquet", world_root),
                    per_stay(base / T13 / script_file, world_root))
        pv, iv = verdicts(x2)
        rows.append({"run": run, "agent": agent, "setting": setting, "world": world_root or "main", "seed": seed,
                     "instruction": message, "x2": x2, "verdict_point": pv, "verdict_interval": iv,
                     "x3": x3, "x0": x0, "total_nll_change": parts[run]["total_nll_per_stay"]["change"],
                     "value_nll_change": e["value_nll_change"], "declined": st.get("refusals", 0),
                     "insulin_unedited": dmax <= 0.01, "insulin_max_change": round(dmax, 4),
                     "no_edit_submitted": st.get("edit_hash") == "none" or e["value_nll_change"] == 0
                     and parts[run]["total_nll_per_stay"]["change"] == 0,
                     "script_x2": script_x2, "paired": pr})

    for run, agent, seed in MAIN:
        g = gen[GEN_KEY[run]]["doses"]
        add(run, agent, "main world, seed 0", None, 0, "main", g["insulin_x3@12"]["ratio"][0], g["insulin_x0@12"]["ratio"][0],
            ctl("scriptS", 0)["insulin_x2"]["ratio"], "shifts_scriptS_s0.parquet")
    for run in TOLD:
        v, s = pa[run], pa[run]["seed"]
        add(run, "Claude" if run.startswith("claude") else "Codex", f"main world, seed {s}", None, s, "told",
            v["unqueried"]["insulin_x3@12"][0], v["unqueried"]["insulin_x0@12"][0], ctl("scriptS", s)["insulin_x2"]["ratio"],
            f"shifts_scriptS_s{s}.parquet")
    for run, v in cs["runs"].items():
        m = v["world"]
        mws.use_world(m)                                          # the truth replay on this world's own held-out stays
        sj = json.loads((ROOT / "results" / "world_install" / f"{m}_script.json").read_text())["held_out"]["readout"]
        add(run, "Claude" if run.startswith("claude") else "Codex", f"world {m[-1]}", m, None,
            "main" if asg[run]["message"] == "message.txt" else "told",
            v["unqueried"]["insulin_x3@12"][0], v["unqueried"]["insulin_x0@12"][0], sj, "shifts_inst_script_r10_s0.parquet")
    out = {"submissions": rows, "apart": {"paused": cs.get("paused", {}), "rerun": list(cs.get("rerun", {}))}}
    (ROOT / "results" / "submissions.json").write_text(json.dumps(out, indent=1) + "\n")
    for r in rows:
        print(f"{r['run']:10s} {r['setting']:20s} {r['instruction']:5s} x2 {r['x2'][0]:.2f} ({r['x2'][1]:.2f}-{r['x2'][2]:.2f}) "
              f"{r['verdict_interval']:9s} x3 {r['x3']:.2f} x0 {r['x0']:.2f} tot {r['total_nll_change']:.3g} dec {r['declined']:3d} "
              f"null {r['insulin_unedited']!s:5s} script {r['script_x2'][0]:.2f} diff {r['paired']['difference']} "
              f"chk {r['paired']['agent']:.2f}/{r['paired']['script']:.2f}")


if __name__ == "__main__":
    main()
