"""The estimator audit along effect strength (the reading fixed before the runs).

For each main-world estimator (Claude runs 1-3, Codex runs 1-3, the full script) and each effect strength beta in
{0, 0.25, 0.5, 1}, on the six paired worlds of each strength (training seeds 101 and 5101-5105; beta 0: the original
no-effect world n03, whose seeds are 101/202, and b000_5101-5105), the error of its implied insulin x2 shift: the value
minus the simulator's true shift on the same stays and stay-hours (tuning stays; training stays for Codex's run 1). Also
the original no-effect set (n03, n4101-n4105) as in the paper.

Per strength: the six worlds' errors, their mean, a stratified interval (one resampled copy per world, 20,000 means) and
a two-stage one (worlds resampled too), each judged against a tenth of the mean true shift at that strength on the scored
stays (at beta 0, where the truth is 0, the main world's band). The design's read-out: a line through the errors at beta
0.25, 0.5 and 1 against the true shift, extrapolated to a true shift of 0, compared with the measured beta-0 error (its
interval resamples one copy per world). And, not in the design (reported as post hoc), the paired difference of Claude's
first estimator minus each of its repeats, draw by draw on the same resampled stays, which removes the noise the
estimators share within a world. Tier occupancy per strength from results/tier_occupancy.json.

    python scripts/graded_summary.py   -> results/graded_summary.json      (CPU, seconds)
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
EA, AE = R / "estimator_audit", R / "audit_eq2"
SEEDS = (101, 5101, 5102, 5103, 5104, 5105)
LEVELS = {0.0: ["n03"] + [f"b000_{s}" for s in SEEDS[1:]],
          0.25: [f"b025_{s}" for s in SEEDS], 0.5: [f"b050_{s}" for s in SEEDS],
          1.0: ["e03"] + [f"e{s}" for s in SEEDS[1:]]}
ORIGINAL_NULL = ["n03", "n4101", "n4102", "n4103", "n4104", "n4105"]
ESTS = ("claude_a", "claude_d", "claude_e", "openai_a", "openai_d", "openai_e", "script")
STAYS = {"openai_a": "train"}                      # every other estimator averages over the tuning stays
N_BOOT = 20_000


def truth(world: str, stays: str) -> float:
    if world.startswith(("n", "b000")):
        return 0.0
    for f in (AE / "truth.json", EA / "truth.json"):
        t = json.loads(f.read_text())
        if world in t:
            return float(t[world][stays])
    raise KeyError(world)


def draws(est: str, world: str) -> dict[int, float]:
    """{draw: implied shift}; draw -1 is the world's own records"""
    if est == "script":
        for f in (EA / f"script_{world}.json", AE / f"script_{world}.json"):
            if f.is_file():
                v = json.loads(f.read_text())["values"]
                return {d - 1: float(x) for d, x in enumerate(v)}
        if world == "n03":
            return {r["draw"]: r["implied_insulin_x2"] for r in map(json.loads, open(R / "replay_bootstrap" / "script_n03.jsonl"))}
        if world.startswith("n4"):
            v = json.loads((R / "fresh_null_seeds_draws.json").read_text())[world[1:]]
            return {d - 1: float(x) for d, x in enumerate(v)}
        return {}
    for f in (EA / f"{est}_{world}.jsonl", AE / f"{est}_{world}.jsonl"):
        if f.is_file():
            rows = [json.loads(x) for x in open(f)]
            out = {r["draw"]: r["eq2"] for r in rows if r.get("eq2") is not None}
            if out:
                return out
    if est == "openai_a" and world in ("n03",) + tuple(f"n{s}" for s in (4101, 4102, 4103, 4104, 4105)):
        f = R / "replay_bootstrap" / "openai_a_n03.jsonl" if world == "n03" else R / "fresh_null_bootstrap" / f"openai_a_{world[1:]}.jsonl"
        return {r["draw"]: r["implied_insulin_x2"] for r in map(json.loads, open(f)) if r.get("implied_insulin_x2") is not None}
    return {}


def errors(est: str, world: str) -> tuple[float | None, np.ndarray]:
    d = draws(est, world)
    t = truth(world, STAYS.get(est, "tuning"))
    pt = d.get(-1)
    return (None if pt is None else pt - t), np.array([v - t for k, v in d.items() if k >= 0])


def verdict(lo, hi, band):
    if -band < lo and hi < band:
        return "within"
    if lo > band or hi < -band:
        return "beyond"
    return "straddles"


def pool(est: str, worlds: list[str], band: float, rng) -> dict:
    pts, cps = [], []
    for w in worlds:
        p, c = errors(est, w)
        if p is None:
            return {"worlds": worlds, "complete": False}
        pts.append(p); cps.append(c)
    out = {"worlds": worlds, "complete": True, "points": pts, "mean": float(np.mean(pts)), "band": band,
           "copies_min": int(min(len(c) for c in cps))}
    if out["copies_min"] >= 10:
        strat = np.mean([rng.choice(c, N_BOOT) for c in cps], axis=0)
        pick = rng.integers(0, len(cps), (N_BOOT, len(cps)))
        two = np.mean([[rng.choice(cps[w]) for w in row] for row in pick[:2000]], axis=1)   # 2,000 two-stage means
        out["interval"] = [float(x) for x in np.percentile(strat, [2.5, 97.5])]
        out["two_stage_interval"] = [float(x) for x in np.percentile(two, [2.5, 97.5])]
        out["verdict"] = verdict(*out["interval"], band)
        out["two_stage_verdict"] = verdict(*out["two_stage_interval"], band)
    return out


def intercept(est: str, rng) -> dict:
    """OLS of error on true shift over the 18 worlds at beta 0.25, 0.5, 1, extrapolated to a true shift of 0"""
    xs, pts, cps = [], [], []
    for b in (0.25, 0.5, 1.0):
        for w in LEVELS[b]:
            p, c = errors(est, w)
            if p is None:
                return {"complete": False}
            xs.append(truth(w, STAYS.get(est, "tuning"))); pts.append(p); cps.append(c)
    X = np.column_stack([np.ones(len(xs)), xs])
    beta = np.linalg.lstsq(X, np.array(pts), rcond=None)[0]
    out = {"complete": True, "intercept": float(beta[0]), "slope": float(beta[1]),
           # the 18 inputs (world, true shift, error), and the same line through the three strengths' means
           "inputs": [[w, x, p] for w, x, p in zip([w for b in (0.25, 0.5, 1.0) for w in LEVELS[b]], xs, pts, strict=True)]}
    xm = [float(np.mean(xs[i:i + 6])) for i in (0, 6, 12)]; pm = [float(np.mean(pts[i:i + 6])) for i in (0, 6, 12)]
    out["intercept_means"] = float(np.linalg.lstsq(np.column_stack([np.ones(3), xm]), np.array(pm), rcond=None)[0][0])
    if min(len(c) for c in cps) >= 10:
        Y = np.column_stack([rng.choice(c, 4000) for c in cps])           # one copy per world per resample
        B = np.linalg.lstsq(X, Y.T, rcond=None)[0]
        out["interval"] = [float(x) for x in np.percentile(B[0], [2.5, 97.5])]
    return out


def paired(a: str, b: str, worlds: list[str], rng) -> dict:
    """a minus b, draw by draw on the same resampled stays (the same draw index resamples the same stays)"""
    pts, cps = [], []
    for w in worlds:
        da, db = draws(a, w), draws(b, w)
        if -1 not in da or -1 not in db:
            return {"complete": False}
        pts.append(da[-1] - db[-1])
        common = sorted((set(da) & set(db)) - {-1})
        cps.append(np.array([da[k] - db[k] for k in common]))
    out = {"complete": True, "points": pts, "mean": float(np.mean(pts)), "copies_min": int(min(len(c) for c in cps))}
    if out["copies_min"] >= 10:
        strat = np.mean([rng.choice(c, N_BOOT) for c in cps], axis=0)
        out["interval"] = [float(x) for x in np.percentile(strat, [2.5, 97.5])]
    return out


def main() -> None:
    occ = json.loads((R / "tier_occupancy.json").read_text())
    res = {"levels": {}, "intercepts": {}, "paired": {}, "original_null": {}, "tier_top_doses": {}}
    mean_truth = {b: float(np.mean([abs(truth(w, "tuning")) for w in LEVELS[b]])) for b in (0.25, 0.5, 1.0)}
    mean_truth_train = {b: float(np.mean([abs(truth(w, "train")) for w in LEVELS[b]])) for b in (0.25, 0.5, 1.0)}
    res["mean_true_shift"] = {"tuning": mean_truth, "train": mean_truth_train}
    for i, est in enumerate(ESTS):
        # each estimator its own seeded stream, so adding one estimator's copies moves no other interval
        rng = np.random.default_rng([20261110, i])
        mt = mean_truth_train if est in STAYS else mean_truth
        res["levels"][est] = {str(b): pool(est, ws, 0.1 * (mt[b] if b > 0 else mt[1.0]), rng) for b, ws in LEVELS.items()}
        res["original_null"][est] = pool(est, ORIGINAL_NULL, 0.1 * mt[1.0], rng)
        # all eleven no-effect worlds: the original six and the five paired ones (b000_5101-5105)
        ten = pool(est, ORIGINAL_NULL + LEVELS[0.0][1:], 0.1 * mt[1.0], rng)
        if ten.get("complete"):
            ten["points_positive"] = int(sum(x > 0 for x in ten["points"]))
        res.setdefault("null_eleven", {})[est] = ten
        res["intercepts"][est] = intercept(est, rng)
    for j, b_ in enumerate(("claude_d", "claude_e", "script")):
        rng = np.random.default_rng([20261110, 100 + j])
        res["paired"][f"claude_a-{b_}"] = {str(b): paired("claude_a", b_, ws, rng) for b, ws in LEVELS.items()}
        res["paired"][f"claude_a-{b_}"]["original_null"] = paired("claude_a", b_, ORIGINAL_NULL, rng)
    # the confounded world: points on six seeds, no copies yet
    CONF = [f"c100_{s}" for s in SEEDS]
    res["confounded"] = {"mean_true_shift": float(np.mean([abs(truth(w, "tuning")) for w in CONF]))}
    for est in ESTS:
        e = [errors(est, w)[0] for w in CONF]
        res["confounded"][est] = {"points": e, "mean": (float(np.mean(e)) if all(x is not None for x in e) else None)}
    for b, key in ((0.0, "b000_"), (0.25, "b025_"), (0.5, "b050_"), (1.0, "e")):
        m = occ.get(f"mean_{key}") if key != "e" else occ.get("mean_e")
        res["tier_top_doses"][str(b)] = None if m is None else m["top_tier_doses"]
    res["tier_top_doses"]["original_null"] = occ["mean_n"]["top_tier_doses"]
    (R / "graded_summary.json").write_text(json.dumps(res, indent=1) + "\n")
    for est in ESTS:
        row = {b: (round(v["mean"], 2) if v.get("complete") else None) for b, v in res["levels"][est].items()}
        ic = res["intercepts"][est]
        print(est, row, "intercept", round(ic["intercept"], 2) if ic.get("complete") else None, ic.get("interval"), flush=True)
    for k, v in res["paired"].items():
        print(k, {b: (round(x["mean"], 2) if x.get("complete") else None, [round(y, 2) for y in x.get("interval", [])]) for b, x in v.items()})


if __name__ == "__main__":
    main()
