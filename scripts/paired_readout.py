"""Paired held-out comparison of installed corrections: the insulin x2 readout of two models on the same 600 held-out
stays, the stays resampled jointly, with the models fixed. This is the readout's own noise; the estimators' sampling
noise is in scripts/paired_bootstrap.py.

    python scripts/paired_readout.py   -> results/paired_readout.json (CPU; the simulator code's Python, for the truth)
Differences are reported as a fraction of the truth (readout minus readout) and in mg/dL (mean shift minus mean shift),
each with a 95% interval, against the band of one tenth of the true effect (0.1 of the truth; 1.4 mg/dL).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import generalize_summary as gs  # noqa: E402

T13 = ROOT / "work" / "runs" / "table13" / gs.WORLD["e03"][0]
METHODS = {"Claude, weight access": "agent_claude_a", "Codex, weight access": "agent_openai_a",
           "script, Claude's estimate": "scriptC", "script, scripted estimate": "scriptS", "fine-tuning": "ft_pnp"}
PAIRS = [("Claude, weight access", "script, scripted estimate"), ("Codex, weight access", "script, scripted estimate"),
         ("Claude, weight access", "script, Claude's estimate"), ("script, Claude's estimate", "script, scripted estimate")]
N, SEED = 2000, 0


def per_stay(method: str) -> pl.DataFrame:
    g = pl.read_parquet(T13 / f"shifts_{method}_s0.parquet").filter(
        (pl.col("edit") == "insulin_x2") & pl.col("reached") & (pl.col("hour") <= 24))
    g = g.join(gs.truth("e03", "insulin_x2", 12), on=["episode", "hour"], how="left")
    return g.group_by("episode").agg(pl.col("shift").sum().alias("m"), pl.col("true_shift").sum().alias("t"),
                                     pl.len().alias("n")).sort("episode")


def main() -> None:
    stays = {k: per_stay(v) for k, v in METHODS.items()}
    ref = stays["Claude, weight access"]
    band = abs(float(ref["t"].sum() / ref["n"].sum())) / 10        # a tenth of the true shift over the scored stay-hours
    out = {"band_mg_dl": round(band, 3), "band_fraction": 0.1, "draws": N, "pairs": {}}
    for a, b in PAIRS:
        j = stays[a].join(stays[b], on="episode", suffix="_b")
        assert (j["n"] == j["n_b"]).all() and np.allclose(j["t"], j["t_b"]), "the two models must share stay-hours"
        ma, mb, t, n = (j[c].to_numpy() for c in ("m", "m_b", "t", "n"))
        pick = np.random.default_rng(SEED).integers(0, len(t), size=(N, len(t)))
        T, C = t[pick].sum(1), n[pick].sum(1)
        dr = ma[pick].sum(1) / T - mb[pick].sum(1) / T
        dm = ma[pick].sum(1) / C - mb[pick].sum(1) / C
        point_r, point_m = ma.sum() / t.sum() - mb.sum() / t.sum(), (ma.sum() - mb.sum()) / n.sum()
        lo, hi = np.quantile(dr, [0.025, 0.975])
        out["pairs"][f"{a} - {b}"] = {
            "readouts": [round(float(ma.sum() / t.sum()), 4), round(float(mb.sum() / t.sum()), 4)],
            "difference_fraction": [round(float(point_r), 4), round(float(lo), 4), round(float(hi), 4)],
            "difference_mg_dl": [round(float(point_m), 3)] + [round(float(x), 3) for x in np.quantile(dm, [0.025, 0.975])],
            "n_stays": len(t), "verdict": ("within the band" if -0.1 < lo and hi < 0.1 else
                                           "beyond the band" if lo > 0.1 or hi < -0.1 else "straddles the band edge")}
    (ROOT / "results" / "paired_readout.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
