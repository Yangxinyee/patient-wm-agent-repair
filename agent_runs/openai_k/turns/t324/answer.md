The current edit passes the first factual batch by only 0.00013 nats, leaving little margin for unseen stays. I’ll recover independent validation IDs and the cumulative record-based targets. Consistent factual fit across the remaining tuning stays would support retaining it; deterioration would require a weaker edit.

```python
from pathlib import Path
import numpy as np, pandas as pd, inspect, dis, io

print("=== TUNING VALIDATION BATCHES ===")
tune_ids284 = sorted(pd.read_csv("/data/tuning.csv", usecols=["stay_id"])["stay_id"].unique().tolist())
tune_batches284 = [tune_ids284[i:i+100] for i in range(0, len(tune_ids284), 100)]
for i, ids in enumerate(tune_batches284):
    print("batch", i, ids)
for k in [24, 25, 27, 28]:
    files = list(Path(f"/work/fwd_{k}").glob("*.csv"))
    if files:
        tab284 = pd.read_csv(files[0])
        print("forward", k, "stay IDs", sorted(tab284.stay_id.unique().tolist()))

print("\n=== RECORD EFFECT TARGETS ===")
for n in ["r132_targets", "r244_targets", "rr_Ntargets", "r267_routes", "r272_trials"]:
    if n in globals():
        o = globals()[n]
        if n in ("r267_routes", "r272_trials"):
            print(n)
            for k, v in o.items():
                if isinstance(v, dict):
                    print(k, {kk: (vv.shape if hasattr(vv, "shape") else repr(vv)[:120])
                              for kk, vv in v.items()})
                else:
                    print(k, type(v).__name__, getattr(v, "shape", None))
        else:
            print(n, repr(o)[:1800])

notes284 = Path("/work/notes.md").read_text()
print("\n=== TARGET/CALIBRATION NOTES ===")
for needle in ["## Run 132:", "## Run 244:", "## Run 266:", "## Run 281:"]:
    j = notes284.find(needle)
    if j >= 0:
        end = notes284.find("\n## ", j + 4)
        print(notes284[j:end if end >= 0 else j+1200][:1400])
lines284 = notes284.splitlines()
for j, line in enumerate(lines284):
    if "approximate tuning rollout targets" in line or "six-lag record-identified" in line:
        print("\n".join(lines284[max(0,j-1):j+4]))

print("\n=== EXACT CURRENT EDIT COMPONENTS ===")
with np.load("/work/map_regularized_trial.npz") as arc:
    base284 = arc["baseline"].copy()
    map284 = arc["map_correction"].copy()
    trial284 = arc["core_b_in"].copy()
original284 = np.asarray(load_original("core.b_in.weight"), dtype=np.float32)
current284 = np.asarray(load("core.b_in.weight"), dtype=np.float32)
print("current matches archive", np.max(np.abs(current284-trial284)))
print("norms", {
    "glucose": float(np.linalg.norm(base284-original284)),
    "map_archive": float(np.linalg.norm(map284)),
    "current_minus_baseline": float(np.linalg.norm(current284-base284))
})
print("MAP scaling against archived component",
      float(np.sum((current284-base284)*map284)/np.sum(map284*map284)))

print("\n=== LOCAL PREDICTION INTERFACE ===")
try:
    print(inspect.getsource(r256_predict)[:3500])
except Exception:
    out284 = io.StringIO()
    dis.dis(r256_predict, file=out284)
    print(out284.getvalue()[:3500])

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 284: prepare independent validation\n"
        "- Forward 27 exact factual delta: +0.00486788045 nats on 4,751 readings; narrow pass on one 100-stay batch.\n"
        "- Recovering disjoint tuning batches, cumulative record targets and exact archived edit components.\n"
        "- No weights changed. Python runs: 284; forward passes: 28.\n"
    )
```