"""Summary of the paired stay bootstrap (scripts/paired_bootstrap.py): per world, each estimator's implied insulin x2 shift
and the paired differences Claude - script (tuning stays) and Codex - script (training stays), each with its 95%
interval, judged by one rule in both worlds: the band of +-1.42 mg/dL (a tenth of the true held-out insulin x2 change).
In the no-effect world the band is around 0 for each estimator; for a difference it is around 0 in both worlds.

    python3 scripts/paired_summary.py   -> results/paired_bootstrap_summary.json
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BAND = 1.42


def verdict(lo: float, hi: float) -> str:
    if -BAND < lo and hi < BAND:
        return "within the band"
    if lo > BAND or hi < -BAND:
        return "beyond the band"
    return "straddles the band edge"


def stat(point: float, v: np.ndarray) -> dict:
    lo, hi = np.percentile(v, [2.5, 97.5])
    return {"point": round(float(point), 3), "interval": [round(float(lo), 3), round(float(hi), 3)],
            "share_within_band": round(float(np.mean(np.abs(v) <= BAND)), 3), "verdict": verdict(lo, hi)}


def main() -> None:
    out = {"band_mg_dl": BAND, "worlds": {}}
    for world in ("e03", "n03"):
        f = ROOT / "results" / "paired_bootstrap" / f"{world}.jsonl"
        if not f.is_file():
            continue
        rows = [json.loads(x) for x in open(f)]
        base = next(r for r in rows if r["draw"] == -1)
        ok = [r for r in rows if r["draw"] >= 0 and None not in (r["claude"], r["codex"], r["script_tuning"], r["script_train"])]
        col = {k: np.array([r[k] for r in ok]) for k in ("claude", "codex", "script_tuning", "script_train")}
        w = {"copies": len(ok), "copies_with_a_failed_estimator": sum(1 for r in rows if r["draw"] >= 0) - len(ok)}
        for k in col:
            w[k] = stat(base[k], col[k])
        w["claude - script_tuning"] = stat(base["claude"] - base["script_tuning"], col["claude"] - col["script_tuning"])
        w["codex - script_train"] = stat(base["codex"] - base["script_train"], col["codex"] - col["script_train"])
        out["worlds"][world] = w
    (ROOT / "results" / "paired_bootstrap_summary.json").write_text(json.dumps(out, indent=1) + "\n")
    for world, w in out["worlds"].items():
        print(world, w["copies"], "copies")
        for k in ("claude", "codex", "script_tuning", "script_train", "claude - script_tuning", "codex - script_train"):
            print(f"  {k:24s} {w[k]['point']:+7.2f}  [{w[k]['interval'][0]:+6.2f}, {w[k]['interval'][1]:+6.2f}]  "
                  f"within band {w[k]['share_within_band']:.0%}  {w[k]['verdict']}")


if __name__ == "__main__":
    main()
