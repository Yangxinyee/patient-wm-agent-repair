"""The told runs: the eight runs told that unqueried doses and every head are scored, read by the rule fixed before the
runs against the six current-brief runs on seed 0.

    python scripts/told_runs_summary.py   -> results/told_runs_summary.json
Inputs: results/agent_eval.jsonl (asked readouts, value NLL), work/runs/general/s+1_b1_e0.3/<run>_s0_{doses,starts}*.parquet
(unqueried), results/model_parts.jsonl (every head, against the run's own seed's plain model), results/generalize.json
(the seed-0 runs). A run not yet scored is listed as missing. CPU, synthetic data.
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "agent")]
NEW = {"claude": ["claude_i", "claude_j", "claude_k", "claude_l"], "openai": ["openai_i", "openai_j", "openai_k", "openai_l"]}
OLD = {"claude": ["claude_a", "claude_d", "claude_e"], "openai": ["openai_a", "openai_d", "openai_e"]}
SEED = json.loads((ROOT / "agent_runs" / "MAPPING.json").read_text())["runs"]
ASKED = ("insulin_x2", "insulin_half", "ne_x2", "ne_half")
UNQ = {"doses": ("insulin_x0@12", "insulin_x1.5@12", "insulin_x3@12", "insulin_x2_lab@12", "ne_x0@12", "ne_x1.5@12",
                 "ne_x3@12", "ne_x2_cuff@12"),
       "starts": ("insulin_x2@6", "insulin_x2@16", "insulin_half@6", "insulin_half@16", "ne_x2@6", "ne_x2@16")}


def t_interval(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return [float(x.mean()) if len(x) else None, None, None]
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [float(x.mean()), float(x.mean() - h), float(x.mean() + h)]


def model_seed(run):
    v = SEED.get(run, {})
    return v.get("model_seed", 0) if isinstance(v, dict) else 0


def unqueried(run):
    import generalize_summary as gs
    gen = ROOT / "work" / "runs" / "general" / "s+1_b1_e0.3"
    out = {}
    for design, keys in UNQ.items():
        files = sorted(gen.glob(f"{run}_s0_{design}*.parquet"))
        if not files:
            continue
        rows = pl.concat([pl.read_parquet(f) for f in files]).unique(subset=["episode", "hour", "edit", "start"], keep="first")
        for k in keys:
            e, st = k.split("@")
            sub = rows.filter(pl.col("start") == int(st))
            r = gs.score(sub, "e03", e, int(st))
            if r and "ratio" in r:
                out[k] = r["ratio"]
    return out


def main():
    ev = {json.loads(x)["method"]: json.loads(x) for x in open(ROOT / "results" / "agent_eval.jsonl")}
    parts = {}
    for x in open(ROOT / "results" / "model_parts.jsonl"):
        r = json.loads(x)
        parts[r["label"]] = r
    g0 = json.loads((ROOT / "results" / "generalize.json").read_text())
    import generalize_summary as gs
    tag2key = {tag: f"{wk}:{label}" for label, wk, method, tag in gs.MODELS}
    def old(run, k):
        v = g0.get(tag2key.get(run, ""), {}).get("doses", {}).get(k)
        return v["ratio"][0] if v and "ratio" in v else None
    res = {"runs": {}, "agents": {}, "missing": []}
    for agent, runs in NEW.items():
        for run in runs:
            e = ev.get(f"agent_{run}")
            if e is None:
                res["missing"].append(run)
                continue
            row = {"seed": model_seed(run), "asked": {q: e[q]["ratio"] for q in ASKED},
                   "value_nll_change": e["value_nll_change"], "unqueried": unqueried(run)}
            p = parts.get(run)
            row["total_nll_change"] = p["total_nll_per_stay"]["change"] if p else None
            row["plain_run"] = p.get("plain_run") if p else None
            res["runs"][run] = row
    for agent, runs in NEW.items():
        have = [r for r in runs if r in res["runs"]]
        a = {"n": len(have), "by_seed": {}}
        def col(rs, f):
            return [f(res["runs"][r]) for r in rs if f(res["runs"][r]) is not None]
        for name, f in (("insulin_x2", lambda r: r["asked"]["insulin_x2"][0]),
                        ("insulin_x3", lambda r: (r["unqueried"].get("insulin_x3@12") or [None])[0]),
                        ("insulin_x0", lambda r: (r["unqueried"].get("insulin_x0@12") or [None])[0]),
                        ("ne_x3", lambda r: (r["unqueried"].get("ne_x3@12") or [None])[0]),
                        ("ne_x0", lambda r: (r["unqueried"].get("ne_x0@12") or [None])[0]),
                        ("total_nll_change", lambda r: r["total_nll_change"])):
            a[name] = t_interval(col(have, f))
            for s in (1, 2):
                rs = [r for r in have if res["runs"][r]["seed"] == s]
                a["by_seed"].setdefault(str(s), {})[name] = t_interval(col(rs, f))
        a["seed0_current_brief"] = {k: [old(r, k) for r in OLD[agent]] for k in ("insulin_x3@12", "insulin_x0@12")}
        x3o, x0o = a["seed0_current_brief"]["insulin_x3@12"], a["seed0_current_brief"]["insulin_x0@12"]
        if a["insulin_x3"][0] is not None and None not in x3o + x0o:
            m3, m0 = a["insulin_x3"][0], a["insulin_x0"][0]
            reach = a["insulin_x3"][2] is not None and a["insulin_x3"][2] >= 0.9
            mv3 = abs(1 - np.mean(x3o)) - abs(1 - m3)          # movement toward 1 of the mean x3 readout
            mv0 = abs(1 - np.mean(x0o)) - abs(1 - m0)
            a["rule1_carries"] = bool(reach or mv3 >= 0.10 or mv0 >= 0.20)
            a["rule2_not"] = bool(min(x3o) <= m3 <= max(x3o) and min(x0o) <= m0 <= max(x0o))
            a["moves"] = {"x3_toward_1": float(mv3), "x0_toward_1": float(mv0), "x3_interval_reaches_band": reach}
        res["agents"][agent] = a
    (ROOT / "results" / "told_runs_summary.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "by_seed"} for k, v in res["agents"].items()}, indent=1))
    print("missing:", res["missing"])


if __name__ == "__main__":
    main()
