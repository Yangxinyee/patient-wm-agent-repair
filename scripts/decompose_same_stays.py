"""Estimate and installation on the same stays.

The first decomposition (decompose_edits.py) read the estimate on the 400 tuning stays and the edit on the 600
held-out stays. Here both are read on the same stays, twice:

  tuning      each final edit rolled out with the agents' own dose_edit tool (agent/fwd_job.py, seed 0) on all 400
              tuning stays, the stays every agent sized its edit on; its insulin x2 readout (mg/dL) against the shift its
              own estimate implies on the same stays (Eq. 2). The two scripts' rows come from their tuning logs
              (results/scripted_baseline.json, the same tool, seed and stays).
  held_out    each estimate's implied shift (Eq. 2) on the 600 held-out stays (the simulator's held-out records, seed
              303, as the agents would read them), against the edit's held-out readout (results/agent_eval.jsonl).

Each agent's lag profile comes from its own original transcript (agent_runs/<run>/tool_calls.jsonl); Codex run 1's
from its fitted effect and persistence (replayed output 23). The profile reproduces decompose_edits.py's tuning value exactly
(checked below). Intervals: bootstrap over stays of the readout, the estimate held fixed (1,000 draws, seed 20261110).

    python scripts/decompose_same_stays.py rollouts   (GPU, about 1 min per edit; writes results/same_stays/<run>/)
    python scripts/decompose_same_stays.py summary    (CPU)  -> results/decompose_same_stays.json
Synthetic data only.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = R / "same_stays"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "agent")); sys.path.insert(0, str(ROOT / "scripts"))
RUNS = ("claude_a", "claude_d", "claude_e", "openai_a", "openai_d", "openai_e")
HELD = ROOT / "work" / "same_stays" / "held_out"


def outputs(run: str) -> dict[int, str]:
    if run == "openai_a":
        return {json.loads(x)["run"]: json.loads(x)["output"] for x in open(ROOT / "agent_runs" / "replay" / "openai_a_e03" / "outputs.jsonl")}
    return {c["run"]: c["output"] for c in map(json.loads, open(ROOT / "agent_runs" / run / "tool_calls.jsonl"))
            if c.get("type") == "python" and c.get("run")}


def profile(run: str) -> np.ndarray:
    o = outputs(run)
    if run == "claude_a":
        psi = np.array([float(v) for v in re.search(r"psi finger\s*\[([^\]]*)\]", o[16]).group(1).split()])
    elif run == "openai_a":
        b, r = (float(x) for x in re.search(r"RD dynamic fit \[effect, persistence\]: \[\s*([-0-9.e]+)\s+([-0-9.e]+)\]", o[23]).groups())
        psi = b * r ** np.arange(12)
    elif run in ("claude_d", "claude_e"):
        import replay_reps_claude as rc
        psi = rc.profile(run, o)
    else:
        import replay_reps_codex as rx
        psi = rx.psi(run, o)
    return np.r_[psi, np.zeros(12 - len(psi))][:12]


def held_out_dir() -> Path:
    """the 600 held-out stays (simulator seed 303) as the agents read records, written as a data folder's tuning.csv"""
    import audit_eq2 as ae
    if not (HELD / "tuning.csv").is_file():
        HELD.mkdir(parents=True, exist_ok=True)
        ae.records(ae.MAIN, 303, 600, 3_030_000_000).write_csv(HELD / "tuning.csv.part", float_precision=4)
        (HELD / "tuning.csv.part").rename(HELD / "tuning.csv")
    return HELD


def rollouts() -> None:
    import torch
    torch.cuda.set_per_process_memory_fraction(0.15)          # hard cap, about 14 GB of the 96 GB card
    import fwd_job
    fwd_job.setup_env("e03")
    stays = sorted(int(s) for b in json.loads((ROOT / "agent_runs" / "claude_a" / "work" / "batches.json").read_text()).values() for s in b)
    assert len(stays) == 400
    for run in RUNS:
        out = OUT / run
        if (out / "summary.json").is_file():
            continue
        out.mkdir(parents=True, exist_ok=True)
        st = json.loads((ROOT / "agent_runs" / run / "state.json").read_text())
        models, touched = fwd_job.load_models(st["model_run"], ["edited"], ROOT / "agent_runs" / run / "submission", st["arm"])
        res = fwd_job.dose_edit(models, "tuning", stays, out, 0)
        res["touched"] = touched
        res["gpu_peak_mib"] = round(torch.cuda.max_memory_reserved() / 2**20)
        (out / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
        print(run, {e: v["mean_shift"] for e, v in res["edited"]["edits"].items()}, flush=True)
        del models
        torch.cuda.empty_cache()


def boot_mean(csv: Path, edit: str, draws: int = 1000) -> tuple[float, float, float]:
    import polars as pl
    f = pl.read_csv(csv).filter((pl.col("edit") == edit) & pl.col("reached"))
    g = f.group_by("stay_id").agg(pl.col("shift").sum().alias("s"), pl.len().alias("n")).sort("stay_id")
    s, n = g["s"].to_numpy(), g["n"].to_numpy().astype(float)
    rng = np.random.default_rng(20261110)
    idx = rng.integers(0, len(s), size=(draws, len(s)))
    b = s[idx].sum(1) / n[idx].sum(1)
    return float(s.sum() / n.sum()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def paired(csv: Path, psi: np.ndarray, tun: dict, draws: int = 1000) -> dict:
    """installation on the tool's own stay-hours: per stay, the summed readout and the summed estimate (Eq. 2's term
    sum_L psi_L * (2 - 1) * u_{t-L}, doses from hour 12) on the same reached insulin x2 rows; ratio of sums, and its
    interval resampling stays with both sides together (seed 20261110)"""
    import polars as pl
    row_of = {int(s_): i for i, s_ in enumerate(tun["stay"].tolist())}
    R = tun["I"]
    f = pl.read_csv(csv).filter((pl.col("edit") == "insulin_x2") & pl.col("reached"))
    est = np.array([sum(psi[L - 1] * R[row_of[int(s_)], int(h) - L] for L in range(1, 13) if int(h) - L >= 12)
                    for s_, h in zip(f["stay_id"].to_list(), f["hour"].to_list(), strict=True)])
    g = f.with_columns(pl.Series("est", est)).group_by("stay_id").agg(pl.col("shift").sum().alias("r"), pl.col("est").sum().alias("e"),
                                                                       pl.len().alias("n")).sort("stay_id")
    r, e, n = g["r"].to_numpy(), g["e"].to_numpy(), g["n"].to_numpy().astype(float)
    rng = np.random.default_rng(20261110)
    idx = rng.integers(0, len(r), size=(draws, len(r)))
    b = r[idx].sum(1) / e[idx].sum(1)
    return {"_draws": b, "readout_rows": float(r.sum() / n.sum()), "estimate_rows": float(e.sum() / n.sum()), "installation": float(r.sum() / e.sum()),
            "installation_ci": [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))], "n_rows": int(n.sum()), "n_stays": len(r)}


def heldout_paired(tag: str, psi: np.ndarray, hel: dict, draws: int = 1000) -> dict:
    """held-out stays: per scored row (insulin x2, reached, glucose read) the edit's shift (work/runs/table13), the
    simulator's truth (truth_held_out.parquet) and Eq. (2)'s term of psi; ratios of sums, intervals resampling stays with
    all three together (seed 20261110)"""
    import polars as pl
    t13 = ROOT / "work" / "runs" / "table13" / "s+1_b1_e0.3"
    f = pl.read_parquet(t13 / f"shifts_{tag}_s0.parquet").filter((pl.col("edit") == "insulin_x2") & pl.col("reached") & pl.col("observed"))
    tr = pl.read_parquet(ROOT / "work" / "synth" / "s+1_b1_e0.3" / "truth_held_out.parquet").filter(pl.col("edit") == "insulin_x2")
    f = f.join(tr.select("episode", "hour", "true_shift"), on=["episode", "hour"], how="inner")
    row_of = {int(s_): i for i, s_ in enumerate(hel["stay"].tolist())}
    R = hel["I"]
    est = np.array([sum(psi[L - 1] * R[row_of[int(e)], int(h) - L] for L in range(1, 13) if int(h) - L >= 12)
                    for e, h in zip(f["episode"].to_list(), f["hour"].to_list(), strict=True)])
    g = f.with_columns(pl.Series("est", est)).group_by("episode").agg(pl.col("shift").sum().alias("r"), pl.col("true_shift").sum().alias("t"),
                                                                       pl.col("est").sum().alias("e")).sort("episode")
    r, t, e = g["r"].to_numpy(), g["t"].to_numpy(), g["e"].to_numpy()
    idx = np.random.default_rng(20261110).integers(0, len(r), size=(draws, len(r)))
    out = {"n_rows": f.height, "n_stays": len(r)}
    for k, (num_, den) in {"estimate_ratio": (e, t), "edit_ratio": (r, t), "installation": (r, e)}.items():
        b = num_[idx].sum(1) / den[idx].sum(1)
        out[k] = [float(num_.sum() / den.sum()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]
        if k == "installation":
            out["_draws"] = b
    return out


def summary() -> None:
    import audit_eq2 as ae
    import scripted_estimate as se
    tun = ae.tuning_panel(ROOT / "agent_data" / "e03" / "data" / "tuning.csv")
    hel = ae.tuning_panel(held_out_dir() / "tuning.csv")
    truth_tun = json.loads((R / "audit_eq2" / "truth.json").read_text())["e03"]["tuning"]
    truth_hel_sim = ae.scored_truth(ae.MAIN, 303, 600)
    old = json.loads((R / "decompose_edits.json").read_text())["runs"]
    ev = {json.loads(x)["run"]: json.loads(x) for x in open(R / "agent_eval.jsonl")}
    truth_hel = ev["claude_a"]["insulin_x2"]["true_shift"][0]           # the held-out truth every readout is scored against
    out = {"truth_tuning": truth_tun, "truth_held_out_sim": truth_hel_sim, "runs": {}}
    for run in RUNS:
        psi = profile(run)
        imp_t = float(se.implied(tun, "I", "glucose_finger", psi, 2.0)[0])
        imp_h = float(se.implied(hel, "I", "glucose_finger", psi, 2.0)[0])
        if run != "openai_a":               # Codex run 1 printed its estimate on the training stays; decompose_edits.py re-applied it too
            assert abs(imp_t - old[run]["implied_tuning"]) < 1e-6, (run, imp_t, old[run]["implied_tuning"])
        pr = paired(OUT / run / "dose_edit_edited.csv", psi, tun)
        hp = heldout_paired(f"agent_{run}", psi, hel)
        m, lo, hi = boot_mean(OUT / run / "dose_edit_edited.csv", "insulin_x2")
        e = ev[run]["insulin_x2"]
        out["runs"][run] = {
            "tuning": {"implied": imp_t, "readout": m, "readout_ci": [lo, hi], "installation": m / imp_t,
                       "installation_ci": sorted([lo / imp_t, hi / imp_t]), "estimate_ratio": imp_t / truth_tun,
                       "edit_ratio": m / truth_tun, "residual_mgdl": m - imp_t, "paired": pr},
            "held_out": {"implied": imp_h, "truth": e["true_shift"][0], "readout": e["mean_shift"][0],
                         "estimate_ratio": imp_h / e["true_shift"][0], "edit_ratio": e["ratio"][0],
                         "edit_ratio_ci": e["ratio"][1:], "installation": e["mean_shift"][0] / imp_h, "paired": hp},
        }
        print(run, json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                               for k, v in out["runs"][run].items()}, default=lambda o: "<draws>"), flush=True)
    sb = json.loads((R / "scripted_baseline.json").read_text())
    kern = {"script_claude_estimate": np.r_[[-2.0, -2.3, -2.0, -1.6, -1.1, -0.9, -0.6, -0.4], np.zeros(4)],     # scripted_tune.CLAUDE
            "script": np.r_[json.loads((R / "scripted" / "estimate_e03.json").read_text())["pairs"]["I->glucose_finger"]["psi"][:8], np.zeros(4)]}
    final = {"script_claude_estimate": "scriptC_e03", "script": "scriptS_e03"}
    for name, key in (("script_claude_estimate", "scripted install, Claude's estimate"), ("script", "scripted install, scripted estimate")):
        t = sb[key]["tuning"]
        rounds = sorted((ROOT / "work" / "scripted" / final[name]).glob("round*/dose_edit_m.csv"))
        # the script with Claude's estimate installs the shape of Claude's rounded kernel, sized to Claude's own tuning
        # readout (-13.415): its estimate is that target, so the kernel is rescaled to it before the paired ratio
        scale = t["targets"]["insulin_x2"] / float(se.implied(tun, "I", "glucose_finger", kern[name], 2.0)[0])
        pr = paired(rounds[-1], kern[name] * scale, tun)
        imp_h = float(se.implied(hel, "I", "glucose_finger", kern[name] * scale, 2.0)[0])
        hp = heldout_paired({"script_claude_estimate": "scriptC", "script": "scriptS"}[name], kern[name] * scale, hel)
        print(name, rounds[-1].parent.name, round(scale, 4), {k: (round(v, 3) if isinstance(v, float) else v) for k, v in pr.items() if k != "_draws"}, flush=True)
        out["runs"][name] = {"tuning": {"implied": t["targets"]["insulin_x2"], "readout": t["readouts"]["insulin_x2"],
                                        "installation": t["readouts"]["insulin_x2"] / t["targets"]["insulin_x2"],
                                        "estimate_ratio": t["targets"]["insulin_x2"] / truth_tun,
                                        "edit_ratio": t["readouts"]["insulin_x2"] / truth_tun,
                                        "residual_mgdl": t["readouts"]["insulin_x2"] - t["targets"]["insulin_x2"], "paired": pr},
                             "held_out": {"implied": imp_h, "truth": truth_hel, "estimate_ratio": imp_h / truth_hel,
                                          "edit_ratio": sb[key]["queried"]["insulin_x2"][0],
                                          "edit_ratio_ci": sb[key]["queried"]["insulin_x2"][1:],
                                          "installation": sb[key]["queried"]["insulin_x2"][0] * truth_hel / imp_h, "paired": hp}}
    # tuning minus held-out installation: the two stay sets are independent, so their draws are paired at random
    for v in out["runs"].values():
        tp, hp = v["tuning"]["paired"], v["held_out"]["paired"]
        d = tp.pop("_draws") - hp.pop("_draws")
        v["drop"] = [float(tp["installation"] - hp["installation"][0]), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]
    (R / "decompose_same_stays.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    {"rollouts": rollouts, "summary": summary}[sys.argv[1]]()
