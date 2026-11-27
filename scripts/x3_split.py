"""The insulin x3 readout split by whether the tripled doses reach the model's top dose bin.

The model reads an insulin rate through fine bins whose last edge is 5 U/h, so every rate above 5 U/h shares bin 16
(scripts/x3_encoding.py). A held-out stay-hour t is "below the top bin" when every insulin rate given from hour 12 to
t - 1, tripled, stays at or under 5 U/h; otherwise "in the top bin". For each correction, the x3 readout (mean predicted
shift over mean true shift on reached stay-hours, 95% interval resampling stays) in each group, from the generalization
rollouts (work/runs/general/<world>/<tag>_s0_doses*.parquet) and the simulator's truth (generalize_summary.truth).

    python scripts/x3_split.py   -> results/x3_split.json      (CPU, a minute; synthetic data only)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import generalize_summary as gs  # noqa: E402  (its truth replay and stay bootstrap; the simulator, unchanged)

import polars as pl  # noqa: E402

from backend import sim  # noqa: E402

GEN = ROOT / "work" / "runs" / "general" / "s+1_b1_e0.3"
TAGS = {"claude_a": "Claude, run 1", "claude_d": "Claude, run 2", "claude_e": "Claude, run 3", "openai_a": "Codex, run 1",
        "openai_d": "Codex, run 2", "openai_e": "Codex, run 3", "scriptS": "full script", "scriptC": "script, Claude's estimate",
        "ft": "fine-tuning", "ftC": "fine-tuning, sized", "plain": "unedited"}
TOP = 5.0


def rows(tag: str) -> pl.DataFrame:
    fs = sorted(GEN.glob(f"{tag}_s0_doses*.parquet"))
    assert fs, tag
    return pl.concat([pl.read_parquet(f) for f in fs]).filter((pl.col("edit") == "insulin_x3") & pl.col("reached"))


def main() -> None:
    _, w = gs.WORLD["e03"]
    s = sim.simulate(w, gs.N, gs.SEED)
    ins = np.asarray(s["INS"])
    row_of = {int(x): i for i, x in enumerate(s["subject"].astype(np.int64))}
    truth = gs.truth("e03", "insulin_x3", 12)
    out = {"top_bin_from": TOP, "methods": {}}
    for tag, lab in TAGS.items():
        g = rows(tag).join(truth, on=["episode", "hour"], how="left")
        assert g["true_shift"].null_count() == 0
        top = np.array([bool((3 * ins[row_of[int(e)], sim.PRE + 12: sim.PRE + int(h)] > TOP).any())
                        for e, h in zip(g["episode"].to_list(), g["hour"].to_list(), strict=True)])
        v, t, subj = g["shift"].to_numpy(), g["true_shift"].to_numpy(), g["subject"].to_numpy()
        r = {"label": lab}
        for name, msk in (("all", np.ones_like(top)), ("below_top_bin", ~top), ("top_bin", top)):
            mv, tv = gs.boot_mean(v[msk], subj[msk]), gs.boot_mean(t[msk], subj[msk])
            # ratio interval: the same stays resampled for model and truth
            s_, inv = np.unique(subj[msk], return_inverse=True)
            tm, tt = np.bincount(inv, v[msk]), np.bincount(inv, t[msk])
            pick = np.random.default_rng(0).integers(0, len(s_), size=(1000, len(s_)))
            b = tm[pick].sum(1) / tt[pick].sum(1)
            r[name] = {"ratio": [float(tm.sum() / tt.sum()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))],
                       "model_shift": mv[0], "true_shift": tv[0], "n_stay_hours": int(msk.sum()), "n_stays": int(len(s_))}
        out["methods"][tag] = r
        print(tag, {k: (round(x["ratio"][0], 3), x["n_stay_hours"]) for k, x in r.items() if k != "label"}, flush=True)
    (ROOT / "results" / "x3_split.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
