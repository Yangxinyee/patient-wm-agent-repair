"""The value-NLL cost of each weakened agent edit at its own estimate.

    python scripts/strength_sweep.py sweep <target> [<target> ...]   (GPU) -> results/strength_sweep/<target>.json
    python scripts/strength_sweep.py export <target> <a>              writes the scaled edit as an agent submission under
                                                                      work/strength_sweep/<target>_a<a>/ for generalize.py
    python scripts/strength_sweep.py summary                           -> results/strength_sweep.json
Synthetic data only; the simulator code is called, never changed.
"""
from __future__ import annotations

import json
import time
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "agent"), str(ROOT / "scripts")]
OUT = ROOT / "results" / "strength_sweep"
WORK = ROOT / "work" / "strength_sweep"
MODEL_RUN = "synth_e03-m-T1-full-long-s0"
W2 = "attr_heads.value_bin.2.weight"
BIN = "core.b_in.weight"
#: per target: run, drug readout, how the drug's part is scaled (see the design file)
TARGETS = {
    "codexA_ins": {"run": "openai_a", "edit": "insulin_x2", "cols": [41, 109]},
    "codexB_ins": {"run": "openai_d", "edit": "insulin_x2", "cols": [107, 109]},
    "codexC_ins": {"run": "openai_e", "edit": "insulin_x2", "bin": "ins"},
    "claudeB_ne": {"run": "claude_d", "edit": "ne_x2", "cols": [124, 142]},
    "codexB_ne": {"run": "openai_d", "edit": "ne_x2", "cols": [48, 171]},
    "codexC_ne": {"run": "openai_e", "edit": "ne_x2", "bin": "ne"},
    # Claude 3 held its insulin gain below its estimate for the budget (its final answer); units 26 and 41
    "claudeC_ins": {"run": "claude_e", "edit": "insulin_x2", "cols": [26, 41]},
}
TOL, MAX_STEPS, A_MAX = 0.02, 4, 40.0


def estimate(name: str) -> float:
    t = TARGETS[name]
    if t["edit"] == "insulin_x2":
        return json.loads((ROOT / "results" / "decompose_same_stays.json").read_text())["runs"][t["run"]]["tuning"]["implied"]
    if name == "codexB_ne":
        return json.loads((ROOT / "results" / "ne_installation.json").read_text())["runs"]["openai_d"]["ne_x2_estimate"]
    if name == "codexC_ne":
        return float(np.mean(json.loads((ROOT / "results" / "ne_installation.json").read_text())["runs"]["openai_e"]["ne_x2_estimate_range"]))
    # Claude 2: Eq. (2) for norepinephrine of the clean lag profile in its notes (work/effects.json, lags 1-8)
    import audit_eq2 as ae
    import scripted_estimate as se
    g = np.array(json.loads((ROOT / "agent_runs" / "claude_d" / "work" / "effects.json").read_text())["gam_ne_clean"], dtype=float)
    psi = np.r_[g, np.zeros(12 - len(g))][:12]
    return float(se.implied(ae.tuning_panel(ROOT / "agent_data" / "e03" / "data" / "tuning.csv"), "N", "map_arterial", psi, 2.0)[0])


def states(name: str):
    """(unedited state dict, submitted state dict, function a -> state dict)"""
    import torch
    import fwd_job
    t = TARGETS[name]
    saved = torch.load(fwd_job.WORK_ROOT / "runs" / MODEL_RUN / "model.pt", map_location="cpu", weights_only=False)
    sd0 = saved["model"]
    st = json.loads((ROOT / "agent_runs" / t["run"] / "state.json").read_text())
    sd1, _ = fwd_job.edited_state({k: v.clone() for k, v in sd0.items()}, ROOT / "agent_runs" / t["run"] / "submission", st["arm"])
    if "cols" in t:
        c = t["cols"]

        def make(a: float) -> dict:
            sd = {k: v.clone() for k, v in sd1.items()}
            sd[W2][:, c] = sd1[W2][:, c] * a
            return sd
    else:
        d = torch.from_numpy(np.load(ROOT / "agent_runs" / "openai_e" / "work" / "baseline77.npz")["delta"]).float()
        D = sd1[BIN].float() - sd0[BIN].float()
        k = float((D * d).sum() / (d * d).sum())
        ins = k * d
        parts = {"ins": ins, "ne": D - ins}

        def make(a: float) -> dict:
            sd = {kk: v.clone() for kk, v in sd1.items()}
            other = parts["ne" if t["bin"] == "ins" else "ins"]
            sd[BIN] = (sd0[BIN].float() + other + a * parts[t["bin"]]).to(sd0[BIN].dtype)
            return sd
    return sd0, sd1, make, saved


def sweep(names: list[str]) -> None:
    import torch
    torch.cuda.set_per_process_memory_fraction(0.15)
    import fwd_job
    fwd_job.setup_env("e03")
    from backend import ModelConfig, Vocab, WorldModel
    stays = sorted(int(s) for b in json.loads((ROOT / "agent_runs" / "claude_a" / "work" / "batches.json").read_text()).values() for s in b)
    assert len(stays) == 400
    OUT.mkdir(parents=True, exist_ok=True)

    def model_of(sd, saved):
        m = WorldModel(ModelConfig(**saved["config"]), Vocab(**saved["vocab"])).cuda().eval()
        m.load_state_dict(sd)
        m.keep_probs = True
        return m

    def nll(model, work: Path) -> float:
        torch.manual_seed(0)
        return fwd_job.factual({"m": model}, "tuning", stays, work)["m"]["value_nll_per_value"]

    for name in names:
        t = TARGETS[name]
        f = OUT / f"{name}.json"
        rec = json.loads(f.read_text()) if f.is_file() else {"target": name, **t, "estimate": estimate(name), "points": []}
        sd0, sd1, make, saved = states(name)
        work = WORK / name
        work.mkdir(parents=True, exist_ok=True)
        if "nll_unedited" not in rec:
            m0 = model_of(sd0, saved)
            rec["nll_unedited"] = nll(m0, work)
            del m0
        done = {p["a"]: p for p in rec["points"]}

        def evaluate(a: float) -> dict:
            a = round(float(a), 4)
            if a in done:
                return done[a]
            m = model_of(make(a), saved)
            w = work / f"a{a}"
            w.mkdir(parents=True, exist_ok=True)
            v_nll = nll(m, w)                  # before the rollout: the rollout leaves its noise hook on the model
            r = fwd_job.dose_edit({"m": m}, "tuning", stays, w, 0)["m"]["edits"]
            m.z_noise = None
            p = {"a": a, "readouts": {e: v["mean_shift"] for e, v in r.items()}, "value_nll": v_nll}
            p["cost"] = round(p["value_nll"] - rec["nll_unedited"], 6)
            del m
            torch.cuda.empty_cache()
            done[a] = p
            rec["points"] = sorted(done.values(), key=lambda q: q["a"])
            f.write_text(json.dumps(rec, indent=1) + "\n")
            print(name, a, p["readouts"][t["edit"]], p["cost"], flush=True)
            # the step's rollout files are kept (no deletion)
            return p

        est = rec["estimate"]
        p0, p1 = evaluate(0.0), evaluate(1.0)
        r = lambda p: p["readouts"][t["edit"]]
        lo, hi = p0, p1
        a = 1.0 + (est - r(p1)) * (1.0 - 0.0) / (r(p1) - r(p0)) if r(p1) != r(p0) else A_MAX
        for _ in range(MAX_STEPS):
            a = float(min(max(a, 0.0), A_MAX))
            p = evaluate(a)
            if abs(r(p) - est) <= TOL * abs(est):
                break
            lo, hi = hi, p
            if r(hi) == r(lo):
                break
            a = hi["a"] + (est - r(hi)) * (hi["a"] - lo["a"]) / (r(hi) - r(lo))
        best = min(rec["points"], key=lambda q: abs(r(q) - est))
        rec["full_strength"] = {"a": best["a"], "readout": r(best), "cost": best["cost"], "within_tol": abs(r(best) - est) <= TOL * abs(est)}
        if best["a"] > 1:          # a midpoint for the shape of the curve
            evaluate((1.0 + best["a"]) / 2)
        f.write_text(json.dumps(rec, indent=1) + "\n")


def export(name: str, a: float) -> Path:
    """the scaled edit as a white-box submission (only the changed tensors), for scripts/generalize.py agent:<dir>"""
    import torch
    from safetensors.torch import save_file
    import fwd_job
    fwd_job.setup_env("e03")
    sd0, _, make, _ = states(name)
    sd = make(a)
    d = WORK / f"{name}_a{a}"
    if d.exists():                         # never overwrite or delete; a new folder instead
        d = WORK / f"{name}_a{a}_{time.strftime('%Y%m%d_%H%M%S')}"
    (d / "submission" / "edited").mkdir(parents=True)
    for k, v in sd.items():
        if not torch.equal(v, sd0[k]):
            save_file({"w": v.contiguous()}, str(d / "submission" / "edited" / f"{k}.safetensors"))
    st = json.loads((ROOT / "agent_runs" / TARGETS[name]["run"] / "state.json").read_text())
    (d / "state.json").write_text(json.dumps({"arm": st["arm"], "world": st["world"], "model_run": st["model_run"], "done": True}) + "\n")
    return d


def summary() -> None:
    out = {"budget": 0.005, "targets": {}}
    for name in TARGETS:
        f = OUT / f"{name}.json"
        if f.is_file():
            out["targets"][name] = json.loads(f.read_text())
    (ROOT / "results" / "strength_sweep.json").write_text(json.dumps(out, indent=1) + "\n")
    for n, v in out["targets"].items():
        print(n, v.get("estimate"), v.get("full_strength"))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "sweep":
        sweep(sys.argv[2:])
    elif cmd == "export":
        print(export(sys.argv[2], float(sys.argv[3])))
    else:
        summary()
