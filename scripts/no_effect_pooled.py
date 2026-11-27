"""Pooled no-effect bias of each estimator over the six no-effect worlds (the original one and the five fresh ones), with a
95% interval, judged by the band rule of the paper (within / beyond / straddles +-1.42 mg/dL).

Every value is the implied insulin x2 shift on the readout's scale, Eq. (2) of the paper. Inputs, one stay bootstrap per
world (draw -1 is the world's own records):
  Claude          results/audit_eq2/claude_a_<world>.jsonl (Eq. 2 on its replayed lag profile, tuning stays)
  Codex           results/replay_bootstrap/openai_a_n03.jsonl, results/fresh_null_bootstrap/openai_a_<seed>.jsonl
                  (its code's own Eq. 2, training stays)
  full script     results/replay_bootstrap/script_n03.jsonl, results/fresh_null_seeds_draws.json (Eq. 2 on its kernel)
Main worlds (the original and five fresh ones, scripts/audit_eq2.py): each estimator's error, its value minus the
simulator's true shift on the same stays (results/audit_eq2/truth.json), pooled the same way.

Pooled bias: the mean of the worlds' implied insulin x2 shifts. The worlds are independent draws of one data-generating
process (same simulator and policy, new seeds), so the spread of the pooled mean is the sampling spread of each world's
estimate; its interval is a stratified bootstrap: each of 20,000 draws takes one resampled copy from every world and
averages them. The between-world spread of the points is reported beside the within-world bootstrap spread as a check of
that assumption, and a two-stage bootstrap (worlds resampled with replacement, then one copy from each) as a conservative
variant that counts the sampling spread twice.

    python3 scripts/no_effect_pooled.py   -> results/no_effect_pooled.json
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
BAND = 1.42
SEEDS = (4101, 4102, 4103, 4104, 4105)
N_BOOT = 20_000


def jsonl(p: Path, key: str = "implied_insulin_x2", minus: float = 0.0) -> tuple[float, np.ndarray]:
    rows = list({r["draw"]: r for r in map(json.loads, open(p))}.values())     # one row per draw (two runs may overlap)
    point = next(r[key] for r in rows if r["draw"] == -1)
    v = np.array([r[key] for r in rows if r["draw"] >= 0 and r.get(key) is not None])
    return float(point) - minus, v - minus


REPEATS = ("claude_d", "claude_e", "openai_d", "openai_e")       # the main-world repeats' estimators
EA = R / "estimator_audit"


def worlds(est: str) -> dict[str, tuple[float, np.ndarray]]:
    if est in REPEATS:
        out = {"original": jsonl(EA / f"{est}_n03.jsonl", "eq2")}
        for s in SEEDS:
            out[str(s)] = jsonl(EA / f"{est}_n{s}.jsonl", "eq2")
        return out
    if est == "claude_a":
        out = {"original": jsonl(R / "audit_eq2" / "claude_a_n03.jsonl", "eq2")}
        for s in SEEDS:
            out[str(s)] = jsonl(R / "audit_eq2" / f"claude_a_n{s}.jsonl", "eq2")
        return out
    out = {"original": jsonl(R / "replay_bootstrap" / f"{est}_n03.jsonl")}
    if est == "script":
        d = json.loads((R / "fresh_null_seeds_draws.json").read_text())
        for s in SEEDS:
            v = np.array(d[str(s)])
            out[str(s)] = (float(v[0]), v[1:])
    else:
        for s in SEEDS:
            out[str(s)] = jsonl(R / "fresh_null_bootstrap" / f"{est}_{s}.jsonl")
    return out


MAIN = ("e03", "e5101", "e5102", "e5103", "e5104", "e5105")


def main_errors(est: str) -> dict[str, tuple[float, np.ndarray]]:
    """value minus truth in each main world, on the stays the estimator averages over"""
    tr = json.loads((R / "audit_eq2" / "truth.json").read_text())
    out = {}
    for w in MAIN:
        if est in REPEATS:
            out[w] = jsonl(EA / f"{est}_{w}.jsonl", "eq2", tr[w]["tuning"])
            continue
        if est == "script":
            v = np.array(json.loads((R / "audit_eq2" / f"script_{w}.json").read_text())["values"]) - tr[w]["tuning"]
            out[w] = (float(v[0]), v[1:])
        else:
            out[w] = jsonl(R / "audit_eq2" / f"{est}_{w}.jsonl", "eq2", tr[w]["tuning" if est == "claude_a" else "train"])
    return out


def verdict(lo: float, hi: float, band: float = BAND) -> str:
    if -band < lo and hi < band:
        return "within"
    if lo > band or hi < -band:
        return "beyond"
    return "straddles"


# each estimator is judged against a tenth of the true main-world shift on the stays its Eq. (2) averages over
# (tuning stays for Claude's estimator and the script's, training stays for Codex's), not the held-out band of 1.42; in a
# main world, that world's own truth; pooled and in the no-effect worlds, the mean over the six main worlds
STAYS = {"claude_a": "tuning", "script": "tuning", "openai_a": "train",
         "claude_d": "tuning", "claude_e": "tuning", "openai_d": "tuning", "openai_e": "tuning"}


def scored_band(est: str, world: str | None = None) -> float:
    tr = json.loads((R / "audit_eq2" / "truth.json").read_text())
    if world is not None:
        return 0.1 * abs(tr[world][STAYS[est]])
    return 0.1 * abs(float(np.mean([tr[w][STAYS[est]] for w in MAIN])))


def pooled(ws: dict, keys: list[str], rng: np.random.Generator, band: float | None = None) -> dict:
    pts = np.array([ws[k][0] for k in keys])
    strat = np.mean([rng.choice(ws[k][1], N_BOOT) for k in keys], axis=0)
    two = np.zeros(N_BOOT)
    pick = rng.integers(0, len(keys), (N_BOOT, len(keys)))
    for j in range(len(keys)):
        two += np.array([rng.choice(ws[keys[w]][1]) for w in pick[:, j]])
    two /= len(keys)
    lo, hi = np.percentile(strat, [2.5, 97.5])
    tlo, thi = np.percentile(two, [2.5, 97.5])
    within_sd = float(np.sqrt(np.mean([np.var(ws[k][1], ddof=1) for k in keys])))
    out = {"worlds": keys, "points": [float((x)) for x in pts], "mean": float(pts.mean()),
            "interval": [float((lo)), float((hi))], "verdict": verdict(lo, hi),
            "share_of_draws_beyond_upper_edge": round(float(np.mean(strat > BAND)), 4),
            "two_stage_interval": [float((tlo)), float((thi))], "two_stage_verdict": verdict(tlo, thi),
            "between_world_sd_of_points": float(pts.std(ddof=1)), "within_world_bootstrap_sd": float(within_sd),
            "points_positive": int((pts > 0).sum()), "points_beyond_band": int((np.abs(pts) > BAND).sum())}
    if band is not None:                          # the verdicts on the scored stays' band
        out["band_scored"] = band
        out["verdict_heldout_band"], out["two_stage_verdict_heldout_band"] = out["verdict"], out["two_stage_verdict"]
        out["verdict"], out["two_stage_verdict"] = verdict(lo, hi, band), verdict(tlo, thi, band)
        out["points_beyond_band"] = int((np.abs(pts) > band).sum())
    return out


def paired_claude_minus_script() -> dict:
    """Claude's estimator minus the full script's, draw by draw: both bootstraps resample the same stays for the same draw
    (rng(10_000 + draw).integers(0, n, n) over the stays in id order; Generator.choice with replacement draws the same
    indices), so each draw's difference is paired. In main worlds both are scored against the same truth on the same tuning
    stays, so the difference of their errors is the difference of their values."""
    def claude(w):
        return {r["draw"]: r["eq2"] for r in map(json.loads, open(R / "audit_eq2" / f"claude_a_{w}.jsonl")) if r.get("eq2") is not None}

    def listed(v):
        return {d - 1: float(x) for d, x in enumerate(v)}               # values[0] is draw -1
    fresh = json.loads((R / "fresh_null_seeds_draws.json").read_text())
    out = {"no_effect": {}, "main": {}}
    for k, w in [("original", "n03")] + [(str(s), f"n{s}") for s in SEEDS]:
        s = ({r["draw"]: r["implied_insulin_x2"] for r in map(json.loads, open(R / "replay_bootstrap" / "script_n03.jsonl"))}
             if k == "original" else listed(fresh[k]))
        out["no_effect"][k] = (claude(w), s)
    for w in MAIN:
        out["main"][w] = (claude(w), listed(json.loads((R / "audit_eq2" / f"script_{w}.json").read_text())["values"]))
    res = {}
    for kind, ws in out.items():
        diffs = {}
        for k, (c, s) in ws.items():
            common = sorted((set(c) & set(s)) - {-1})
            diffs[k] = (c[-1] - s[-1], np.array([c[d] - s[d] for d in common]))
        res[kind] = diffs
    return res


def main() -> None:
    rng = np.random.default_rng(20261108)
    out = {"band_mg_dl": BAND, "draws": N_BOOT, "estimators": {}}
    def complete(r: str) -> bool:      # a repeat enters once every world has its point and at least 20 resampled copies
        fs = [EA / f"{r}_{w}.jsonl" for w in ["n03"] + [f"n{s}" for s in SEEDS] + list(MAIN)]
        return all(f.is_file() and sum(json.loads(x).get("eq2") is not None for x in open(f)) >= 21 and
                   any(json.loads(x)["draw"] == -1 and json.loads(x).get("eq2") is not None for x in open(f)) for f in fs)
    ests = ["claude_a", "openai_a", "script"] + [r for r in REPEATS if complete(r)]
    for est in ests:                   # the repeats come after the first three, so their random draws are unchanged
        if est in REPEATS:             # each repeat its own generator, so adding one never moves another
            rng = np.random.default_rng(20261110 + REPEATS.index(est))
        ws = worlds(est)
        per = {}
        for k, (p, v) in ws.items():
            lo, hi = np.percentile(v, [2.5, 97.5])
            per[k] = {"point": float(p), "copies": int(len(v)), "interval": [float((lo)), float((hi))],
                      "sd": float(v.std(ddof=1)), "verdict": verdict(lo, hi, scored_band(est)),
                      "verdict_heldout_band": verdict(lo, hi)}
        me = main_errors(est)
        mpw = {}
        for k, (p, v) in me.items():
            lo, hi = np.percentile(v, [2.5, 97.5])
            bw = scored_band(est, k)
            mpw[k] = {"error": float(p), "copies": int(len(v)), "interval": [float(lo), float(hi)], "band_scored": bw,
                      "verdict": verdict(lo, hi, bw), "point_beyond_band": bool(abs(p) > bw)}
        out["estimators"][est] = {"band_scored": scored_band(est),
                                  "per_world": per,
                                  "pooled_all_six": pooled(ws, list(ws), rng, scored_band(est)),
                                  "pooled_fresh_five": pooled(ws, [str(s) for s in SEEDS], rng, scored_band(est)),
                                  "main_per_world": mpw,
                                  "main_points_beyond_band": int(sum(v["point_beyond_band"] for v in mpw.values())),
                                  "main_pooled": pooled(me, list(me), rng, scored_band(est)),
                                  "excluding_zero": int(sum((lo > 0 or hi < 0) for lo, hi in
                                                            (np.percentile(v, [2.5, 97.5]) for _, v in ws.values())))}
        print(est, json.dumps({k: v for k, v in out["estimators"][est]["pooled_all_six"].items() if k != "worlds"}), flush=True)
        print(est, "main", json.dumps({k: v for k, v in out["estimators"][est]["main_pooled"].items() if k != "worlds"}), flush=True)
        print(est, "fresh", json.dumps({k: v for k, v in out["estimators"][est]["pooled_fresh_five"].items() if k != "worlds"}), flush=True)
    # the paired difference Claude minus the full script, per world and pooled (its own generator, so the
    # numbers above are unchanged)
    prng = np.random.default_rng(20261109)
    out["paired_claude_minus_script"] = {}
    for kind, ws in paired_claude_minus_script().items():
        p = pooled(ws, list(ws), prng)
        p["copies_per_world"] = {k: int(len(v)) for k, (_, v) in ws.items()}
        p["excludes_zero"] = bool(p["interval"][0] > 0 or p["interval"][1] < 0)
        p["two_stage_excludes_zero"] = bool(p["two_stage_interval"][0] > 0 or p["two_stage_interval"][1] < 0)
        out["paired_claude_minus_script"][kind] = p
        print("paired Claude - script", kind, json.dumps({k: v for k, v in p.items() if k not in ("worlds",)}), flush=True)
    (R / "no_effect_pooled.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
