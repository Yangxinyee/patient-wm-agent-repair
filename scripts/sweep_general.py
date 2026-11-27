"""Codex's insulin edits at full strength, read on held-out stays at x3 and x0.

For each insulin target of scripts/strength_sweep.py: the edit scaled to its full-strength factor is exported as a
submission (strength_sweep.py export), rolled out by scripts/generalize.py (design `doses`, seed 0), and its insulin x3
and x0 readouts computed against the simulator's truth (generalize_summary.truth), ratio of means with 1,000 stay
resamples. The submitted edits' readouts (a = 1) are in results/generalize.json.

    python scripts/sweep_general.py run       (GPU; generalize.py per target)
    python scripts/sweep_general.py summary   -> results/sweep_general.json
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "agent")]
PY = sys.executable                     # the simulator code's Python (run this script with it)
TARGETS = ("codexA_ins", "codexB_ins", "codexC_ins")
GEN = ROOT / "work" / "runs" / "general" / "s+1_b1_e0.3"


def full_a(t: str) -> float:
    return json.loads((ROOT / "results" / "strength_sweep" / f"{t}.json").read_text())["full_strength"]["a"]


def run() -> None:
    for t in TARGETS:
        a = full_a(t)
        d = subprocess.run([PY, str(ROOT / "scripts" / "strength_sweep.py"), "export", t, str(a)], check=True,
                           capture_output=True, text=True).stdout.strip().splitlines()[-1]
        st = json.loads((Path(d) / "state.json").read_text())
        subprocess.run([PY, str(ROOT / "scripts" / "generalize.py"), "e03", st["model_run"], f"agent:{d}", f"sweep_{t}", "0", "doses"],
                       check=True)


def summary() -> None:
    import polars as pl
    import generalize_summary as gs
    out = {}
    for t in TARGETS:
        f = GEN / f"sweep_{t}_s0_doses.parquet"
        if not f.is_file():
            continue
        rows = pl.read_parquet(f)
        r = {"a": full_a(t)}
        for e in ("insulin_x3", "insulin_x0"):
            g = rows.filter((pl.col("edit") == e) & pl.col("reached")).join(gs.truth("e03", e, 12), on=["episode", "hour"], how="left")
            v, tt, subj = g["shift"].to_numpy(), g["true_shift"].to_numpy(), g["subject"].to_numpy()
            s_, inv = np.unique(subj, return_inverse=True)
            tm, tr = np.bincount(inv, v), np.bincount(inv, tt)
            pick = np.random.default_rng(0).integers(0, len(s_), size=(1000, len(s_)))
            b = tm[pick].sum(1) / tr[pick].sum(1)
            r[e] = [float(tm.sum() / tr.sum()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]
        out[t] = r
        print(t, r, flush=True)
    (ROOT / "results" / "sweep_general.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    {"run": run, "summary": summary}[sys.argv[1]]()
