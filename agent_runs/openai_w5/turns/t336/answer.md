The earlier record analysis supports glucose-lowering insulin effects and MAP-raising norepinephrine effects, with provisional rollout targets of about −11.9/+5.9 and +4.5/−2.2 respectively. Those estimates remain assumption-dependent. The displayed output cut off before the factual scores, so I’ll retrieve those and the edit controls compactly; a passing score would justify testing stronger insulin response, not accepting it yet.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import torch

print("=== RECENT FACTUAL RESULTS ===")
for d in ["fwd_25", "fwd_28", "fwd_30", "fwd_31", "fwd_32", "fwd_33"]:
    directory = Path("/work") / d
    if not directory.exists():
        continue
    for p in sorted(directory.glob("factual_*.csv")):
        x = pd.read_csv(p)
        nc = [c for c in x if "nll" in c.lower()]
        vals = {c: round(float(x[c].mean()), 7)
                for c in nc if pd.api.types.is_numeric_dtype(x[c])}
        print(d, p.name, len(x), vals)
        if d == "fwd_33":
            print("columns:", x.columns.tolist())
            event_col = next((c for c in ["event", "code", "event_code", "name"] if c in x), None)
            if event_col and nc:
                print(x.groupby(event_col)[nc].mean().round(6).to_string())

print("\n=== TUNING IDS ===")
print(Path("/data/README.md").read_text()[:2400])
tuning_recovery = pd.read_csv("/data/tuning.csv")
print("columns:", tuning_recovery.columns.tolist())
stay_col_recovery = next(c for c in ["stay_id", "stay", "id"] if c in tuning_recovery)
tuning_ids_recovery = sorted(tuning_recovery[stay_col_recovery].unique().tolist())
print("n:", len(tuning_ids_recovery), "first:", tuning_ids_recovery[:8])
print("second block:", tuning_ids_recovery[100:200])

print("\n=== EDIT-CONTROL VARIABLES ===")
snap_recovery = list(globals().items())
for k, v in sorted(snap_recovery):
    if k.startswith("_") or not any(
        word in k.lower() for word in
        ["base264", "base285", "center", "strength", "pair", "coord",
         "private", "neuron", "control", "target", "cols", "idx", "tau"]
    ):
        continue
    if isinstance(v, dict):
        detail = repr(list(v.keys())[:15])
    elif isinstance(v, (list, tuple)):
        detail = repr(v[:12])
    elif isinstance(v, (int, float, str, np.number)):
        detail = repr(v)
    elif callable(v):
        detail = "callable"
    else:
        detail = str(getattr(v, "shape", None))
    print(k, type(v).__name__, detail[:160])

print("\n=== AVAILABLE LOCAL HELPERS ===")
print(", ".join(sorted(
    k for k, v in snap_recovery
    if callable(v) and getattr(v, "__module__", None) == "__main__"
))[:2400])

print("\n=== VALUE-HEAD EDIT LAYOUT ===")
for name in ["attr_heads.value_bin.0.weight", "attr_heads.value_bin.2.weight"]:
    a = torch.as_tensor(load(name)).float()
    b = torch.as_tensor(load_original(name)).float()
    delta = a - b
    print(name, tuple(a.shape))
    if name.endswith("2.weight"):
        norms = torch.linalg.vector_norm(delta, dim=0)
        inds = torch.topk(norms, min(16, len(norms))).indices.tolist()
        print("largest edited columns:", [(i, round(float(norms[i]), 4)) for i in inds])

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 288: compact recovery\n"
        "- Earlier conditional estimates support insulin beta about -1.83 "
        "and NE beta about +3.2 to +3.6; no medical priors used.\n"
        "- Provisional tuning-A AR targets: insulin -11.893/+5.946, "
        "NE +4.497/-2.249. These are not observed intervention truth.\n"
        "- Retrieving latest factual scores and exact edit controls; no weights changed.\n"
    )
```