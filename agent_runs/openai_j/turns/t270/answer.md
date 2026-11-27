The record-based kernel estimates support stronger responses than the original model, but the norepinephrine estimate rests on a stronger identification assumption. Current insulin rollouts slightly overshoot their targets. I’ll check whether reducing insulin strength can free factual-fit budget for MAP; that approach is useful only if both changes generalize to held-out stays.

```python
import dis
import numpy as np
import pandas as pd
from pathlib import Path

print("EXACT MAP HEAD CONSTRUCTION")
print(dis.Bytecode(head177_make).dis()[:3700])

print("\nCACHE AND HELPER STRUCTURE")
for name in ["head177_eval", "affine234", "curve_eval236", "contrast235"]:
    fn = globals()[name]
    print(name, "globals:", fn.__code__.co_names,
          "locals:", fn.__code__.co_varnames)
for name in ["train236", "curve236"]:
    obj = globals()[name]
    print(name)
    for k, v in obj.items():
        if isinstance(v, dict):
            print(" ", k, {kk: getattr(vv, "shape", type(vv).__name__)
                            for kk, vv in v.items()})
        else:
            print(" ", k, getattr(v, "shape", type(v).__name__))

print("\nINSULIN/HEAD HELPERS")
print([k for k, v in dict(globals()).items()
       if callable(v) and not k.startswith("_")
       and any(s in k.lower() for s in
               ["insulin", "trial158", "head16", "head17", "head15"])])

print("\nGPU FACTUAL FILES")
for folder in ["/work/fwd_22", "/work/fwd_25"]:
    paths = list(Path(folder).glob("factual_*.csv"))
    dfs = {p.stem.replace("factual_", ""): pd.read_csv(p) for p in paths}
    print(folder, {k: (v.shape, list(v.columns)) for k, v in dfs.items()})
    if "original" in dfs and "edited" in dfs:
        a, b = dfs["original"], dfs["edited"]
        nc = next((c for c in a.columns if "nll" in c.lower()), None)
        ec = next((c for c in ["event", "code", "event_code"]
                   if c in a.columns), None)
        if nc and ec and len(a) == len(b) and a[ec].equals(b[ec]):
            tmp = pd.DataFrame({"event": a[ec], "delta": b[nc] - a[nc]})
            print(tmp.groupby("event")["delta"].agg(["count", "mean"]))

print("\nINSULIN CHECKPOINT")
with np.load("/work/best_insulin_v1.npz") as z:
    print({k: z[k].shape for k in z.files})
print("MAP free neurons:", head177_free)

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 240: independent-strength audit\n"
        "- Insulin rollout modestly overshoots the record-based kernel target; MAP doubling remains too small.\n"
        "- Examining sensor-specific factual costs and independent head scaling before installing a candidate.\n"
        "- No weights changed; causal target uncertainty remains explicit.\n"
    )
```