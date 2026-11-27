"""Every scored submission's doubled-insulin readout in hours 13-16 and 21-24 separately (ratio of summed
predicted to summed true shift over the reached stay-hours in that window; 95% interval resampling stays, 2,000 draws),
and the same for the full script on the same trained model; whether any lies within the band in both windows.

    python scripts/byhour_submissions.py   -> results/byhour_submissions.json   (CPU; the simulator code's Python)
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

import more_worlds_summary as mws  # noqa: E402

gs = mws.gs
T13 = "runs/table13/s+1_b1_e0.3"
WIN = {"early": (13, 16), "late": (21, 24)}


def hu(x):
    return Decimal(repr(float(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def within(r):
    return Decimal("0.9") <= hu(r[1]) and hu(r[2]) <= Decimal("1.1")


def window(path, lo, hi):
    g = pl.read_parquet(path).filter((pl.col("edit") == "insulin_x2") & pl.col("reached") & (pl.col("hour") >= lo)
                                     & (pl.col("hour") <= hi))
    g = g.join(gs.truth("e03", "insulin_x2", 12), on=["episode", "hour"], how="left")
    s = g.group_by("episode").agg(pl.col("shift").sum().alias("m"), pl.col("true_shift").sum().alias("t")).sort("episode")
    m, t = s["m"].to_numpy(), s["t"].to_numpy()
    pick = np.random.default_rng(0).integers(0, len(t), size=(2000, len(t)))
    b = m[pick].sum(1) / t[pick].sum(1)
    return [round(float(m.sum() / t.sum()), 4)] + [round(float(x), 4) for x in np.quantile(b, [0.025, 0.975])]


def main():
    subs = json.loads((ROOT / "results" / "submissions.json").read_text())["submissions"]
    out, done_worlds = {"submissions": {}, "scripts": {}}, set()
    for r in sorted(subs, key=lambda r: r["world"] != "main"):          # the main world first, then each further world
        base = ROOT / "work" if r["world"] == "main" else ROOT / "work" / "fresh_main" / r["world"]
        if r["world"] != "main":
            mws.use_world(r["world"])
        o = {k: window(base / T13 / f"shifts_agent_{r['run']}_s0.parquet", *w) for k, w in WIN.items()}
        o["both_within"] = within(o["early"]) and within(o["late"])
        out["submissions"][r["run"]] = o
        sk = f"main_s{r['seed']}" if r["world"] == "main" else r["world"]
        if sk not in done_worlds:
            f = (f"shifts_scriptS_s{r['seed']}.parquet" if r["world"] == "main" else "shifts_inst_script_r10_s0.parquet")
            so = {k: window(base / T13 / f, *w) for k, w in WIN.items()}
            so["both_within"] = within(so["early"]) and within(so["late"])
            out["scripts"][sk] = so
            done_worlds.add(sk)
    (ROOT / "results" / "byhour_submissions.json").write_text(json.dumps(out, indent=1) + "\n")
    for k, v in list(out["submissions"].items()) + list(out["scripts"].items()):
        print(f"{k:12s} early {v['early']} late {v['late']} both {v['both_within']}")


if __name__ == "__main__":
    main()
