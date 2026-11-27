"""Ten new no-effect worlds against the two earlier sets (the reading fixed before the new worlds were run).

For each of the seven estimators: its point (world's own records) on the six earlier worlds (n03, n4101-n4105), the five
on the graded seeds (b000_5101-5105) and the ten new ones (b000_6101-6110); the three set means; how many new points lie
above the earlier six's lowest and below the five's highest; the design's case 1 / case 2 test; the 21-world mean with
a world-resampled interval (20,000 resamples of the 21 points), judged against the band (+-1.33 mg/dL).

    python scripts/null_seed_batch.py   -> results/null_seed_batch.json      (CPU, seconds)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import graded_summary as gs  # noqa: E402

SIX = gs.ORIGINAL_NULL
FIVE = [f"b000_{s}" for s in (5101, 5102, 5103, 5104, 5105)]
TEN = [f"b000_{6101 + i}" for i in range(10)]
BAND = 1.33


def points(est: str, worlds: list[str]) -> list[float | None]:
    return [gs.errors(est, w)[0] for w in worlds]


def main() -> None:
    rng = np.random.default_rng(20261110)
    out = {"band": BAND, "sets": {"six": SIX, "five": FIVE, "ten": TEN}, "estimators": {}}
    for est in gs.ESTS:
        six, five, ten = points(est, SIX), points(est, FIVE), points(est, TEN)
        r = {"six": six, "five": five, "ten": ten}
        if any(v is None for v in six + five + ten):
            r["complete"] = False
            out["estimators"][est] = r
            print(est, "incomplete", flush=True)
            continue
        m6, m5, m10 = float(np.mean(six)), float(np.mean(five)), float(np.mean(ten))
        allp = np.array(six + five + ten)
        boot = allp[rng.integers(0, len(allp), (20_000, len(allp)))].mean(1)
        lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
        r.update({"complete": True, "mean_six": m6, "mean_five": m5, "mean_ten": m10,
                  "ten_between_set_means": bool(min(m6, m5) <= m10 <= max(m6, m5)),
                  "ten_above_six_min": int(sum(v > min(six) for v in ten)),
                  "ten_below_five_max": int(sum(v < max(five) for v in ten)),
                  "ten_above_mean_six": int(sum(v > m6 for v in ten)), "ten_below_mean_five": int(sum(v < m5 for v in ten)),
                  "ten_sd": float(np.std(ten, ddof=1)), "six_sd": float(np.std(six, ddof=1)), "five_sd": float(np.std(five, ddof=1)),
                  "mean_21": float(allp.mean()), "interval_21": [lo, hi], "verdict_21": gs.verdict(lo, hi, BAND),
                  "mean_11": float(np.mean(six + five))})
        # the design's case 2: the ten sit with one earlier set (all ten above the five's mean, or all ten below the six's)
        r["sits_with_one_set"] = bool(all(v > m5 for v in ten) or all(v < m6 for v in ten)) if m6 > m5 else \
            bool(all(v < m5 for v in ten) or all(v > m6 for v in ten))
        out["estimators"][est] = r
        print(est, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("six", "five", "ten")}, flush=True)
    # post hoc (labelled so in the paper): the first six against the other fifteen, exact one-sided permutation
    # p of the six's mean exceeding the rest's, over all C(21, 6) = 54,264 splits
    from itertools import combinations
    for est in ("claude_a", "claude_d", "claude_e", "script"):
        v = np.array(points(est, SIX + FIVE + TEN), dtype=float)
        obs = v[:6].mean() - v[6:].mean()
        tot, cnt, n = v.sum(), 0, 0
        for c in combinations(range(len(v)), 6):
            m6 = v[list(c)].mean()
            if m6 - (tot - 6 * m6) / 15 >= obs - 1e-12:
                cnt += 1
            n += 1
        out["estimators"][est]["six_minus_rest"] = float(obs)
        out["estimators"][est]["perm_p_six_high"] = cnt / n
        print(est, "six - rest", round(obs, 3), "p", round(cnt / n, 5), flush=True)
    # Claude run 1's excess over its repeats, world by world (points), and how many of the 21 worlds it reads positive
    W = SIX + FIVE + TEN
    a = np.array(points("claude_a", W))
    out["claude_a_positive_21"] = int((a > 0).sum())
    for b in ("claude_d", "claude_e"):
        d = a - np.array(points(b, W))
        bt = d[rng.integers(0, len(d), (20_000, len(d)))].mean(1)
        out[f"excess_claude_a_minus_{b}"] = {"mean_21": float(d.mean()), "interval_21": [float(x) for x in np.percentile(bt, [2.5, 97.5])],
                                              "mean_ten": float(d[-10:].mean()), "mean_eleven": float(d[:11].mean()),
                                              "mean_six": float(d[:6].mean()), "mean_five": float(d[6:11].mean()),
                                              "positive_21": int((d > 0).sum())}
        print(b, out[f"excess_claude_a_minus_{b}"])
    four = ("claude_a", "claude_d", "claude_e", "script")
    done = [e for e in four if out["estimators"][e].get("complete")]
    if len(done) == 4:
        case2 = sum(out["estimators"][e]["sits_with_one_set"] for e in four) >= 3
        out["reading"] = "case 2: a seed-range effect" if case2 else "case 1: world-level spread"
        print(out["reading"])
    (ROOT / "results" / "null_seed_batch.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
