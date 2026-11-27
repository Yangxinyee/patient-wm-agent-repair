"""Same-shape references (rules fixed before the runs): the full script's sizing, but the estimate installed by
same-shape tensor replacement only, as the agents' tool allows.

    python scripts/same_shape.py dirs                    # each drug's direction in core.b_in's input (2,000 training stays)
    python scripts/same_shape.py tune <variant> <estimate>   # variant bdir | ncol; estimate script | claude
Variants: bdir, core.b_in.weight <- W (I + (g_I - 1) v_I v_I^T + (g_N - 1) v_N v_N^T) (orthonormal directions); ncol,
embed.numeric.weight columns z_dose and z_rate times one gain g. Sizing as scripts/scripted_tune.py: the agents' dose_edit
tool code on the 400 tuning stays, at most 4 secant rounds per drug (bdir) or on insulin x2 (ncol), stop within 3%;
then the factual check (value NLL per reading, tuning stays); above 0.005 the gains' excess over 1 is scaled by 0.9, 0.8,
... until within it. The scored model is written as an agent-style submission (agent_runs/ss_<variant>_<estimate>/:
submission/edited/<tensor>.safetensors and state.json) so it is scored by the agents' own pipeline.
-> results/same_shape/tune_<variant>_<estimate>.json. GPU (waits for <= 80 C before each pass). Synthetic data only.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "agent"), str(ROOT / "scripts")]
import backend  # noqa: E402

backend.configure(mhat_file="atoms.mhat_prev.npy")
import fwd_job  # noqa: E402

KEY, PLAIN = "e03", "synth_e03-m-T1-full-long-s0"
fwd_job.setup_env(KEY)
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from scripted_estimate import LAGS, panel  # noqa: E402

OUT = ROOT / "work" / "same_shape"
RES = ROOT / "results" / "same_shape"
STAYS = sorted(int(s) for b in json.loads((ROOT / "agent_runs" / "claude_a" / "work" / "batches.json").read_text()).values() for s in b)
assert len(STAYS) == 400
#: as scripts/scripted_tune.py: Claude's estimate (lag profiles, its four tuning-stay targets) and the unedited model's
#: tuning readouts (Claude's tool, the 400 tuning stays, original weights)
CLAUDE = {"kernels": {"I": [-2.0, -2.3, -2.0, -1.6, -1.1, -0.9, -0.6, -0.4], "N": [3.4, 2.6, 1.75, 0.9, 0.4, 0.2]},
          "targets": {"insulin_x2": -13.415, "insulin_half": 6.707, "ne_x2": 4.474, "ne_half": -2.237}}
PLAIN_TUNING = {"insulin_x2": -1.80, "insulin_half": 0.82, "ne_x2": -0.23, "ne_half": -0.56}
DRUG_EDITS = {"I": ("insulin_x2", "insulin_half"), "N": ("ne_x2", "ne_half")}
BUDGET = 0.005


def cool():
    while int(subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.split()[0]) > 80:
        time.sleep(30)


def plain_sd():
    return torch.load(ROOT / "work" / "runs" / PLAIN / "model.pt", map_location="cpu", weights_only=False)["model"]


def directions(n_stays: int = 2000) -> None:
    from backend import evaluation as oe
    from backend import WindowSpec, make_loader
    from backend import load_actions
    cool()
    root, ids = oe.store_root(), oe.code_ids()
    ins, ne = ids["SOURCE/inputevents/223258"], ids["OMOP/1321341"]
    stays = sorted(pl.read_csv(ROOT / "agent_data" / KEY / "data" / "train.csv", columns=["stay_id"])["stay_id"].unique().to_list())[:n_stays]
    models, _ = fwd_job.load_models(PLAIN, ["original"], OUT, "white")
    model = models["original"]
    spec = WindowSpec(window=128, effect_fields=False, **load_actions(oe.COHORT, root))
    rec = []
    hook = model.core.b_in.register_forward_pre_hook(lambda mod, inp: rec.append(inp[0].detach().float().mean(1)))
    loader = make_loader(root, "train", spec=spec, batch_size=64, shuffle=False, num_workers=0,
                         episode_filter=lambda ep: ep.filter(pl.col("subject_id").is_in(stays)))
    sums = {k: np.zeros(192) for k in ("ins", "no_ins", "ne", "no_ne")}
    cnt = {k: 0 for k in sums}
    state = None
    with torch.no_grad():
        for b in loader:
            B, K = b["bundle_mask"].shape
            if state is None or state["h"].shape[0] != B:
                state = model.init_state(B, torch.device("cuda"))
            b = {k: v.cuda() for k, v in b.items()}
            rec.clear()
            _, state = model(b, state)
            e = torch.stack(rec, 1).cpu().numpy()                      # (B, K, 192): one b_in call per bundle step
            assert e.shape[1] == K, (e.shape, K)
            mask = b["bundle_mask"].cpu().numpy()
            code, kind, bundle = (b[k].cpu().numpy() for k in ("atom_code", "atom_kind", "atom_bundle"))
            has = {c: np.zeros(B * K, bool) for c in (ins, ne)}
            for c in (ins, ne):
                has[c][bundle[(code == c) & (kind == 0)]] = True
            for lb in range(B):
                for k in np.flatnonzero(mask[lb]):
                    g = lb * K + k
                    for name, c in (("ins", ins), ("ne", ne)):
                        key = name if has[c][g] else "no_" + name
                        sums[key] += e[lb, k]
                        cnt[key] += 1
    hook.remove()
    vI = sums["ins"] / cnt["ins"] - sums["no_ins"] / cnt["no_ins"]
    vN = sums["ne"] / cnt["ne"] - sums["no_ne"] / cnt["no_ne"]
    vI /= np.linalg.norm(vI)
    vN = vN - (vN @ vI) * vI
    vN /= np.linalg.norm(vN)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "dirs.npz", vI=vI, vN=vN, counts=np.array([cnt[k] for k in ("ins", "no_ins", "ne", "no_ne")]))
    print("directions", {k: cnt[k] for k in cnt}, "cos(vI, raw vN)", round(float(vI @ (sums["ne"] / cnt["ne"] - sums["no_ne"] / cnt["no_ne"])), 4))


MATCHED = {"mclaude": "claude_a", "mcodex": "openai_d"}      # the in-band runs' own writes


def run_delta(run: str) -> dict:
    """the run's submitted same-shape tensors minus the unedited ones, for every tensor it replaced"""
    from safetensors.torch import load_file
    sd = plain_sd()
    return {f.name[: -len(".safetensors")]: load_file(str(f))["w"].double() - sd[f.name[: -len(".safetensors")]].double()
            for f in sorted((ROOT / "agent_runs" / run / "submission" / "edited").glob("*.safetensors"))}


def edit_tensors(variant: str, g: dict) -> dict:
    sd = plain_sd()
    if variant in MATCHED:                             # W(g) = W_0 + g (W_r - W_0)
        return {k: (sd[k].double() + g["I"] * d).to(sd[k].dtype) for k, d in run_delta(MATCHED[variant]).items()}
    if variant == "bdir":
        d = np.load(OUT / "dirs.npz")
        vI, vN = torch.from_numpy(d["vI"]).double(), torch.from_numpy(d["vN"]).double()
        P = torch.eye(192, dtype=torch.float64) + (g["I"] - 1) * torch.outer(vI, vI) + (g["N"] - 1) * torch.outer(vN, vN)
        W = sd["core.b_in.weight"].double()
        return {"core.b_in.weight": (W @ P).to(sd["core.b_in.weight"].dtype)}
    W = sd["embed.numeric.weight"].clone()
    W[:, 2] *= g["I"]
    W[:, 4] *= g["I"]
    return {"embed.numeric.weight": W}


def write(edits: dict, work: Path) -> None:
    d = work / "edited"
    d.mkdir(parents=True, exist_ok=True)
    for name, t in edits.items():
        save_file({"w": t.contiguous()}, str(d / f"{name}.safetensors"))


def readouts(work: Path) -> dict:
    cool()
    models, _ = fwd_job.load_models(PLAIN, ["edited"], work, "white")
    res = fwd_job.dose_edit({"m": models["edited"]}, "tuning", STAYS, work, 0)
    del models
    torch.cuda.empty_cache()
    return {e: v["mean_shift"] for e, v in res["m"]["edits"].items()}


def row_targets(csv: Path, kernels: dict) -> dict:
    """the estimate's readout on exactly the stay-hours the tool pools (as scripts/scripted_tune.py)"""
    p = panel(ROOT / "agent_data" / KEY / "data" / "tuning.csv")
    row_of = {s: i for i, s in enumerate(p["stay"].tolist())}
    f = pl.read_csv(csv).filter(pl.col("reached"))
    out = {}
    for e, (drug, fac) in {"insulin_x2": ("I", 2.0), "insulin_half": ("I", 0.5), "ne_x2": ("N", 2.0), "ne_half": ("N", 0.5)}.items():
        psi = np.zeros(LAGS)
        psi[: len(kernels[drug])] = kernels[drug]
        g = f.filter(pl.col("edit") == e)
        R = p[drug]
        vals = [sum(psi[L - 1] * (fac - 1) * R[row_of[int(s)], int(h) - L] for L in range(1, LAGS + 1) if int(h) - L >= 12)
                for s, h in zip(g["stay_id"].to_list(), g["hour"].to_list(), strict=True)]
        out[e] = float(np.mean(vals))
    return out


def factual(work: Path) -> float:
    cool()
    models, _ = fwd_job.load_models(PLAIN, ["original", "edited"], work, "white")
    fo = work / "factual"
    fo.mkdir(parents=True, exist_ok=True)
    res = fwd_job.factual(models, "tuning", STAYS, fo)
    del models
    torch.cuda.empty_cache()
    return round(res["edited"]["value_nll_per_value"] - res["original"]["value_nll_per_value"], 5)


def tune(variant: str, est: str) -> None:
    if est == "claude":
        kernels, targets = CLAUDE["kernels"], dict(CLAUDE["targets"])
    else:
        e = json.loads((ROOT / "results" / "scripted" / f"estimate_{KEY}.json").read_text())
        kernels, targets = {"I": e["pairs"]["I->glucose_finger"]["psi"][:8], "N": e["pairs"]["N->map_arterial"]["psi"][:8]}, None
    tag = f"{variant}_{est}"
    RES.mkdir(parents=True, exist_ok=True)
    log = {"variant": variant, "estimate": est, "kernels": kernels, "rounds": []}
    rec = RES / f"tune_{tag}.json"
    t0 = time.time()
    if variant in MATCHED:                               # gain 0 is the unedited model, gain 1 the run's own edit
        hist = [({"I": 0.0, "N": 0.0}, dict(PLAIN_TUNING))]
        g = {"I": 1.0, "N": 1.0}
    else:
        hist = [({"I": 1.0, "N": 1.0}, dict(PLAIN_TUNING))]          # gain 1 is the unedited model
        g = {"I": 4.0, "N": 4.0} if variant == "bdir" else {"I": 3.0, "N": 1.0}
    for rnd in range(4):
        work = OUT / tag / f"round{rnd}"
        write(edit_tensors(variant, g), work)
        R = readouts(work)
        if targets is None:
            targets = row_targets(work / "dose_edit_m.csv", kernels)
            log["targets_from"] = "estimate on the tool's reached rows, round 0"
        hist.append((dict(g), R))
        log["rounds"].append({"gains": dict(g), "readouts": R, "targets": targets, "minutes": round((time.time() - t0) / 60, 1)})
        log["targets"] = targets
        rec.write_text(json.dumps(log, indent=1))
        print(f"{tag} round {rnd}: gains {g} readouts {R} targets {targets}", flush=True)
        keys = ("insulin_x2", "ne_x2") if variant == "bdir" else ("insulin_x2",)
        if all(abs(R[e] - targets[e]) <= 0.03 * abs(targets[e]) + 0.05 for e in keys):
            break
        (g0, R0), (g1, R1) = hist[-2], hist[-1]
        new = dict(g1)
        for d, es in (DRUG_EDITS.items() if variant == "bdir" else (("I", ("insulin_x2",)),)):
            u = np.array([(R1[x] - R0[x]) for x in es]) / (g1[d] - g0[d]) if g1[d] != g0[d] else None
            if u is None or not np.isfinite(u).all() or (u ** 2).sum() < 1e-9:
                continue
            T = np.array([targets[x] for x in es])
            base = np.array([R1[x] for x in es]) - g1[d] * u
            new[d] = float(np.clip((u @ (T - base)) / (u @ u), -50, 50))
        g = new
    else:
        log["converged"] = False
    log.setdefault("converged", True)
    gfit = hist[-1][0]
    log["fitted_gains"] = gfit
    # the factual check; above the budget, scale the gains' excess over 1 down together
    final = None
    for f in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
        gs = {d: (f * gfit[d] if variant in MATCHED else 1 + f * (gfit[d] - 1)) for d in gfit}
        work = OUT / tag / f"budget_{f:.1f}"
        write(edit_tensors(variant, gs), work)
        dn = factual(work)
        log.setdefault("budget_steps", []).append({"scale": f, "gains": gs, "value_nll_change": dn})
        rec.write_text(json.dumps(log, indent=1))
        print(f"{tag} scale {f}: gains {gs} value NLL change {dn}", flush=True)
        if dn <= BUDGET:
            final = (f, gs, work)
            break
    if final is None:
        log["final"] = None
        print(f"{tag}: no scale within the budget", flush=True)
    else:
        f, gs, work = final
        log["final"] = {"scale": f, "gains": gs}
        if f < 1.0:                                     # the tuning readouts of the model actually scored
            log["final"]["readouts"] = readouts(work)
        sub = ROOT / "agent_runs" / f"ss_{tag}"
        write(edit_tensors(variant, gs), sub / "submission")
        (sub / "state.json").write_text(json.dumps(
            {"name": f"ss_{tag}", "agent": "same-shape script", "world": KEY, "arm": "white", "model_run": PLAIN, "done": True,
             "passes": 0, "runs": 0, "gains": gs}, indent=1))
    log["minutes"] = round((time.time() - t0) / 60, 1)
    rec.write_text(json.dumps(log, indent=1))
    print(f"{tag} done in {log['minutes']} min; final {log['final']}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "dirs":
        directions(int(sys.argv[2]) if len(sys.argv) > 2 else 2000)
    else:
        tune(sys.argv[2], sys.argv[3])
