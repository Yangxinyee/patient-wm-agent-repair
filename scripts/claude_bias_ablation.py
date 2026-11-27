"""Where Claude's main-world estimator gets its no-effect bias: its own Python runs 1-17 replayed in the sandbox (as
scripts/audit_eq2.py), once unchanged and once per change of its design, on the six no-effect and six main worlds of the
audit; each replay's implied insulin x2 shift is Eq. (2) on the world's tuning stays (as audit_eq2.one).

Claude's main-world estimator (runs 10, 15, 16) regresses later finger-stick glucose on the insulin rate among given doses at
a reading of at least 150, with the current reading (linear) and tier dummies as controls, and deconvolves the later doses.
Its tiers are cut at 150, 180, 220 and 250, while the sliding scale steps at 150, 180, 220, 260 and 300 (App. A): its top tier
holds three base doses (3, 4 and 6 U/h), so within it the dose still rises with glucose beyond the linear control. Readings
exactly on a cut-point get a dose of either neighbouring tier (the doctors act on the unrounded value). In its no-effect run,
Claude built a different estimator (cells of one mg/dL, cut-point readings dropped; claude_b runs 40-49).

Variants (text substitutions in runs 10 and 15; every other run is replayed unchanged):
  as_run      unchanged (must reproduce results/audit_eq2/claude_a_<world>.jsonl, draw -1)
  scale_tiers its tiers cut where the sliding scale steps (150, 180, 220, 260, 300): five tier dummies
  drop_cuts   readings exactly on a cut-point of the scale (150, 180, 220, 260, 300) left out of the regression sample
  both        scale_tiers and drop_cuts

    python3 scripts/claude_bias_ablation.py [--workers 6]   -> results/claude_bias_ablation.json
CPU only; synthetic data only; the simulator is called and never changed.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "scripts"))

WORK = ROOT / "work" / "claude_bias_ablation"
OUT = ROOT / "results" / "claude_bias_ablation.json"
LAST = 17
CUTS = (150, 180, 220, 260, 300)
OLD_TIER = ("H['tier'] = np.select([H.glucose_finger>=250, H.glucose_finger>=220, H.glucose_finger>=180, "
            "H.glucose_finger>=150],[4,3,2,1], 0)")
NEW_TIER = ("H['tier'] = np.select([H.glucose_finger>=300, H.glucose_finger>=260, H.glucose_finger>=220, "
            "H.glucose_finger>=180, H.glucose_finger>=150],[5,4,3,2,1], 0)")
OLD_SAMPLE = "base_ins = H[(H.tier>=1)&(H.I>0)].copy()"
DROP = f"base_ins = base_ins[~base_ins.glucose_finger.isin({list(CUTS)})].copy()"


def variant_code(code: dict, v: str) -> dict:
    c = dict(code)
    if v in ("scale_tiers", "both"):
        assert OLD_TIER in c[10]
        c[10] = c[10].replace(OLD_TIER, NEW_TIER)
        assert "for t_ in [1,2,3,4]:" in c[15] and "X = ['glucose_finger','T2','T3','T4','I']" in c[15]
        c[15] = c[15].replace("for t_ in [1,2,3,4]:", "for t_ in [1,2,3,4,5]:")
        c[15] = c[15].replace("X = ['glucose_finger','T2','T3','T4','I']", "X = ['glucose_finger','T2','T3','T4','T5','I']")
    if v in ("drop_cuts", "both"):
        assert OLD_SAMPLE in c[15]
        c[15] = c[15].replace(OLD_SAMPLE, OLD_SAMPLE + "\n" + DROP)
    return c


def one(args) -> dict:
    v, name = args
    import audit_eq2 as ae
    import scripted_estimate as se
    from agent.kernel import Kernel
    data = ae.data_dir(name)
    calls = [json.loads(x) for x in open(ROOT / "agent_runs" / "claude_a" / "tool_calls.jsonl")]
    code = {c["run"]: c["code"] for c in calls if c.get("type") == "python" and c.get("run") and c["run"] <= LAST}
    code = variant_code(code, v)
    work = WORK / f"work_{v}_{name}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    k = Kernel(work, ROOT / "agent_data" / ae.worlds()[name][3] / "model", data, "white")
    outputs = {}
    try:
        for r in sorted(code):
            outputs[r] = k.run(code[r])[0]
    finally:
        k.kill()
    row = {"variant": v, "world": name}
    try:
        psi = np.array([float(x) for x in re.search(r"psi finger\s*\[([^\]]*)\]", outputs[16]).group(1).split()])
        row["psi"] = psi.tolist()
        row["eq2"] = se.implied(ae.tuning_panel(data / "tuning.csv"), "I", "glucose_finger", psi, 2.0)[0]
        row["error"] = None
    except Exception as e:  # noqa: BLE001
        row["eq2"], row["error"] = None, f"{type(e).__name__}: {e}"
    shutil.rmtree(work, ignore_errors=True)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    import audit_eq2 as ae
    names = [n for n in ae.worlds()]
    jobs = [(v, n) for v in ("as_run", "scale_tiers", "drop_cuts", "both") for n in names]
    rows = []
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex:
        for row in ex.map(one, jobs):
            rows.append(row)
            print(row["variant"], row["world"], row["eq2"], row["error"] or "", flush=True)
    truth = json.loads((ROOT / "results" / "audit_eq2" / "truth.json").read_text())
    ref = {}
    for n in names:
        f = ROOT / "results" / "audit_eq2" / f"claude_a_{n}.jsonl"
        ref[n] = next(json.loads(x)["eq2"] for x in open(f) if json.loads(x)["draw"] == -1)
    out = {"note": __doc__.split("\n\n")[0], "cuts": CUTS, "rows": rows, "audit_as_run": ref, "truth_tuning": {k: v["tuning"] for k, v in truth.items()}}
    summ = {}
    for v in ("as_run", "scale_tiers", "drop_cuts", "both"):
        r = {x["world"]: x["eq2"] for x in rows if x["variant"] == v}
        nulls = [r[n] for n in names if n.startswith("n") and r.get(n) is not None]
        errs = [r[n] - truth[n]["tuning"] for n in names if n.startswith("e") and r.get(n) is not None]
        summ[v] = {"no_effect_mean": float(np.mean(nulls)), "no_effect_min": float(np.min(nulls)), "no_effect_max": float(np.max(nulls)),
                   "main_error_mean": float(np.mean(errs)), "main_error_min": float(np.min(errs)), "main_error_max": float(np.max(errs))}
    out["summary"] = summ
    asrun = {x["world"]: x["eq2"] for x in rows if x["variant"] == "as_run"}
    out["as_run_reproduces_audit"] = all(abs(asrun[n] - ref[n]) < 1e-9 for n in names)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(summ, indent=1), "\nreproduces audit:", out["as_run_reproduces_audit"])


if __name__ == "__main__":
    main()
