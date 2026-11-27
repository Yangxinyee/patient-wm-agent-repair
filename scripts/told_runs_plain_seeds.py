"""The told runs: the unedited models of training seeds 0, 1 and 2 on the unasked doses, the start each told
edit is measured from. Seed 0: tag plain (generalize.json's plain model); seed 1: claude_i, which submitted the unedited
seed-1 model (value and total NLL change 0); seed 2: tag plainS2 (rollout seed 0, as for the
told runs). All in the original main world, scored against its held-out truth.

    python scripts/told_runs_plain_seeds.py   -> results/told_runs_plain_seeds.json
CPU, synthetic data.
"""
import json
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "agent")]
import generalize_summary as gs  # noqa: E402

GEN = ROOT / "work" / "runs" / "general" / "s+1_b1_e0.3"
TAGS = {"0": "plain", "1": "claude_i", "2": "plainS2"}
EDITS = ("insulin_x3", "insulin_x0", "insulin_x1.5", "ne_x3", "ne_x0")


def main():
    gs.check_truth()
    out = {"seeds": {}}
    for seed, tag in TAGS.items():
        files = sorted(GEN.glob(f"{tag}_s0_doses*.parquet"))
        rows = pl.concat([pl.read_parquet(f) for f in files]).unique(subset=["episode", "hour", "edit", "start"], keep="first")
        out["seeds"][seed] = {"tag": tag, **{f"{e}@12": gs.score(rows.filter(pl.col("start") == 12), "e03", e, 12)["ratio"]
                                             for e in EDITS}}
    (ROOT / "results" / "told_runs_plain_seeds.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({s: {k: round(v[0], 3) for k, v in d.items() if k != "tag"} for s, d in out["seeds"].items()}))


if __name__ == "__main__":
    main()
