"""The further main worlds: ten weight-access runs, one Claude and one Codex in each further main world m5101-m5105,
read by the rule fixed before the runs.

    python scripts/more_worlds_summary.py            -> results/more_worlds_summary.json
    python scripts/more_worlds_summary.py --check    (verify each world's truth replay against its stored truth, nothing else)
Inputs: results/agent_eval.jsonl (asked readouts, value NLL; scored in each run's own world), results/model_parts.jsonl
(every head), <world root>/runs/general/s+1_b1_e0.3/<run>_s0_{doses,starts}*.parquet (unasked), results/world_install/
<world>_script.json (each world's script readout), and each world's unedited model (agent_eval method
unedited_<world>, unasked tag "unedited"), a descriptive baseline added after the rule was fixed. The unasked doses' truth is replayed on the world's own
held-out stays (its world.json: 600 stays, seed 5301-5305), checked against the world's stored truth first. A run not yet
scored is listed as missing. CPU, synthetic data.
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "agent")]
import generalize_summary as gs  # noqa: E402

A = json.loads((ROOT / "logs" / "more_worlds" / "assignment.json").read_text())
OLD = {"claude": ["claude_a", "claude_d", "claude_e"], "openai": ["openai_a", "openai_d", "openai_e"]}
ASKED = ("insulin_x2", "insulin_half", "ne_x2", "ne_half")
UNQ = {"doses": ("insulin_x0@12", "insulin_x1.5@12", "insulin_x3@12", "insulin_x2_lab@12", "ne_x0@12", "ne_x1.5@12",
                 "ne_x3@12", "ne_x2_cuff@12"),
       "starts": ("insulin_x2@6", "insulin_x2@16", "insulin_half@6", "insulin_half@16", "ne_x2@6", "ne_x2@16")}
COHORT = "s+1_b1_e0.3"


def verdict(r):
    lo, hi = r[1], r[2]
    return "within" if lo >= 0.9 and hi <= 1.1 else ("beyond" if hi < 0.9 or lo > 1.1 else "straddles")


def t_interval(x):
    x = np.asarray([v for v in x if v is not None], float)
    if len(x) < 2:
        return [float(x.mean()) if len(x) else None, None, None]
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [float(x.mean()), float(x.mean() - h), float(x.mean() + h)]


_simulate = gs.sim.simulate
HO_BASE = 3_030_000_000          # scripts/fresh_main_world.py: a further world's held-out stay ids are moved to this range


def _simulate_renumbered(world, n, seed, *a, **k):
    s = _simulate(world, n, seed, *a, **k)
    if 5301 <= seed <= 5305:
        s["subject"] = s["subject"] - seed * 10_000_000 + HO_BASE
    return s


gs.sim.simulate = _simulate_renumbered


def use_world(m):
    """point generalize_summary's truth replay at world m's own held-out stays, and check it against the stored truth"""
    root = ROOT / "work" / "fresh_main" / m
    wj = json.loads((root / "synth" / COHORT / "world.json").read_text())
    gs.N, gs.SEED = wj["splits"]["held_out"]
    gs._TRUTH.clear()
    ref = pl.read_parquet(root / "synth" / COHORT / "truth_held_out.parquet").filter(pl.col("edit") == "insulin_x2")
    mine = gs.truth("e03", "insulin_x2", 12).join(ref.select("episode", "hour", pl.col("true_shift").alias("ref")),
                                                  on=["episode", "hour"])
    assert mine.height == ref.height and float((mine["true_shift"] - mine["ref"]).abs().max()) < 1e-9, f"{m}: truth replay mismatch"
    return root


def unqueried(run, root):
    gen = root / "runs" / "general" / COHORT
    out = {}
    for design, keys in UNQ.items():
        files = sorted(gen.glob(f"{run}_s0_{design}*.parquet"))
        if not files:
            continue
        rows = pl.concat([pl.read_parquet(f) for f in files]).unique(subset=["episode", "hour", "edit", "start"], keep="first")
        for k in keys:
            e, st = k.split("@")
            r = gs.score(rows.filter(pl.col("start") == int(st)), "e03", e, int(st))
            if r and "ratio" in r:
                out[k] = r["ratio"]
    return out


def scored_run(run, agent, world, message, e, p, root):
    st = json.loads((ROOT / "agent_runs" / run / "state.json").read_text())
    return {"agent": agent, "world": world, "message": message,
            "asked": {q: e[q]["ratio"] for q in ASKED}, "verdict_x2": verdict(e["insulin_x2"]["ratio"]),
            "value_nll_change": e["value_nll_change"], "unqueried": unqueried(run, root),
            "total_nll_change": p["total_nll_per_stay"]["change"],
            "no_edit": st.get("edit_hash") == "none",            # submitted the unedited model
            "usd": st.get("usd"), "runs": st.get("runs"), "passes": st.get("passes"),
            "refusals": st.get("refusals", 0), "refusal_restarts": st.get("refusal_restarts", 0), "turns": st.get("turn")}


#: the rerun of claude_w3 (its rule fixed before it ran): reported apart from the ten, never pooled with them
RERUN = {"claude_w3r": {"agent": "claude", "world_root": "m5103", "message": "message_w3r.txt", "rerun_of": "claude_w3"}}


def main(check_only=False):
    worlds = sorted({r["world_root"] for r in A["runs"]})
    if check_only:
        for m in worlds:
            use_world(m)
            print(m, "truth replay matches", gs.N, gs.SEED)
        return
    ev = {json.loads(x)["method"]: json.loads(x) for x in open(ROOT / "results" / "agent_eval.jsonl")}
    parts = {json.loads(x)["label"]: json.loads(x) for x in open(ROOT / "results" / "model_parts.jsonl")}
    res = {"runs": {}, "agents": {}, "worlds": {}, "missing": []}
    for m in worlds:
        sj = ROOT / "results" / "world_install" / f"{m}_script.json"
        res["worlds"][m] = {"script_readout": json.loads(sj.read_text())["held_out"]["readout"] if sj.is_file() else None}
    for m in worlds:
        root = use_world(m)
        u = ev.get(f"unedited_{m}")                   # descriptive baseline added after the rule was fixed
        res["worlds"][m]["unedited"] = ({"asked": {q: u[q]["ratio"] for q in ASKED}, "unqueried": unqueried("unedited", root)}
                                        if u else None)
        for r in [x for x in A["runs"] if x["world_root"] == m]:
            run = r["run"]
            e = ev.get(f"agent_{run}")
            p = parts.get(run)
            if e is None or p is None:
                res["missing"].append(run)
                continue
            res["runs"][run] = scored_run(run, r["agent"], m, r["message"], e, p, root)
    for run in res["missing"]:                                # a run paused before it submitted (claude_w3)
        st = json.loads((ROOT / "agent_runs" / run / "state.json").read_text())
        if st.get("stops") and not st.get("done"):
            res.setdefault("paused", {})[run] = {"after_turn": st["stops"][-1]["after_turn"], "usd": st.get("usd"),
                                                 "runs": st.get("runs"), "passes": st.get("passes"),
                                                 "refusals": st.get("refusals", 0), "submitted": st.get("submitted")}
    res["rerun"] = {}
    for run, r in RERUN.items():
        root = use_world(r["world_root"])
        e, p = ev.get(f"agent_{run}"), parts.get(run)
        if e is None or p is None:
            res["missing"].append(run)
            continue
        res["rerun"][run] = {**scored_run(run, r["agent"], r["world_root"], r["message"], e, p, root),
                             "rerun_of": r["rerun_of"]}
    for agent in ("claude", "openai"):
        have = {k: v for k, v in res["runs"].items() if v["agent"] == agent}
        f3 = lambda v: (v["unqueried"].get("insulin_x3@12") or [None])[0]       # noqa: E731
        f0 = lambda v: (v["unqueried"].get("insulin_x0@12") or [None])[0]       # noqa: E731
        a = {"n": len(have),
             "verdicts_x2": {w: sum(v["verdict_x2"] == w for v in have.values()) for w in ("within", "straddles", "beyond")},
             "insulin_x2": t_interval([v["asked"]["insulin_x2"][0] for v in have.values()]),
             "insulin_x3": t_interval([f3(v) for v in have.values()]),
             "insulin_x0": t_interval([f0(v) for v in have.values()]),
             "total_nll_change": t_interval([v["total_nll_change"] for v in have.values()]),
             "seed0_main_world_x2": [ev[f"agent_{r}"]["insulin_x2"]["ratio"] for r in OLD[agent]],
             "by_instruction": {}}
        for msg in ("message.txt", "message_unqueried.txt"):                     # descriptive, no test (design page)
            sub = [v for v in have.values() if v["message"] == msg]
            a["by_instruction"][msg] = {"n": len(sub), "insulin_x2": t_interval([v["asked"]["insulin_x2"][0] for v in sub]),
                                        "insulin_x3": t_interval([f3(v) for v in sub]),
                                        "insulin_x0": t_interval([f0(v) for v in sub]),
                                        "total_nll_change": t_interval([v["total_nll_change"] for v in sub])}
        res["agents"][agent] = a
    (ROOT / "results" / "more_worlds_summary.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("by_instruction", "seed0_main_world_x2")}
                      for k, v in res["agents"].items()}, indent=1))
    print("missing:", res["missing"])


if __name__ == "__main__":
    main(check_only="--check" in sys.argv)
