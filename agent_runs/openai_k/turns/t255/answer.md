The records support an immediate glucose decrease of about 2 units per insulin-rate unit and an arterial MAP increase of about 3 per norepinephrine-rate unit, with effects persisting across later hours. The new replay runs, but that alone does not validate it. I’ll compare it directly with official original states; close agreement would justify using it to design a recurrent correction.

```python
from pathlib import Path
import numpy as np

print("=== CAPTURE METADATA ===")
print({k: brief226(v) for k, v in captures151.items()})
print("indices:", indices151[:8])
print("state keys:", [state_keys_audit[int(i)] for i in indices151[:8]])
print("original_result151:", brief226(original_result151))

print("=== OFFICIAL STATE FILES ===")
npz229 = {}
for m229 in ("original", "edited"):
    p229 = Path("/work/fwd_17") / ("states_" + m229 + ".npz")
    if p229.exists():
        npz229[m229] = np.load(p229, allow_pickle=False)
        print(m229, {k: (npz229[m229][k].shape, str(npz229[m229][k].dtype))
                     for k in npz229[m229].files})
        for k in npz229[m229].files:
            a229 = npz229[m229][k]
            if a229.ndim == 1:
                print(k, a229[:5])
            elif a229.ndim == 2 and a229.shape[-1] == 992:
                print(k, "RMS out/h/z:",
                      [float(np.sqrt(np.mean(a229[:, sl] ** 2)))
                       for sl in (slice(0,192), slice(192,960), slice(960,992))])

print("=== EXISTING REPLAY COMPARISON ===")
if isinstance(original_result151, tuple):
    for a229 in original_result151:
        if hasattr(a229, "shape") and tuple(a229.shape) == tuple(base228["out"].shape):
            print("local original out difference RMS:",
                  float(torch.as_tensor(a229).sub(base228["out"]).square().mean().sqrt()))
        if hasattr(a229, "shape") and tuple(a229.shape) == tuple(base228["h"].shape):
            print("local original h difference RMS:",
                  float(torch.as_tensor(a229).sub(base228["h"]).square().mean().sqrt()))
print("=== BLOCK INSTRUCTIONS ===")
print(compact227(audit_block)[-2900:])
if "rr33_scores" in globals():
    print("best prior core diagnostics:", sorted(rr33_scores, key=lambda x: x[0])[:3])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 229: validate local core against official states\n"
            "- Record-based immediate and multi-hour effects recovered; no medical assumptions used.\n"
            "- Inspecting official state files and matching captured contexts before any core edit.\n"
            "- No weights changed. Python runs: 229; forward passes: 17.\n")
```