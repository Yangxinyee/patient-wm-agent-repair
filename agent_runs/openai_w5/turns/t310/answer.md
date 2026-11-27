The recovered diagnostics do not establish the cause of the asymmetry. Persistence helped the insulin response but hurt factual fit, so I’ll test whether its exposure centering has become miscalibrated. I’ll compare factual prediction biases and prepare fresh GPU states for refitting that centering. Recovery of factual fit without losing the dose response would support retaining persistence.

```python
import numpy as np, pandas as pd, torch
from pathlib import Path
from safetensors.torch import load_file
print("=== TRIAL FACTUAL FILE ===")
f270=pd.read_csv("/work/fwd_25/factual_edited.csv")
fo270=pd.read_csv("/work/fwd_25/factual_original.csv")
print(f270.columns.tolist())
print(f270.head(2).to_string(index=False))
print("=== HEAD DATA ===")
print(head_rows.columns.tolist())
print(head_rows.head(2).to_string(index=False))
print("head_prev",head_prev[:8])
print("head helper references:",make_head_inputs212.__code__.co_names)
print("=== STATE OBJECTS ===")
for k,v in list(globals().items()):
    if re.search(r"(^sf|^s4$|state|npz)",k,re.I) and not k.startswith("_"):
        desc = str(getattr(v,"shape",""))
        if hasattr(v,"files"): desc=str([(x,v[x].shape) for x in v.files])
        elif isinstance(v,dict): desc=str(list(v)[:12])
        print(k,type(v).__name__,desc[:600])
print("=== GPU STATE FILES ===")
for k in (3,5,13,15,17,19,21):
    p=Path(f"/work/fwd_{k}")
    if p.exists():
        for q in p.glob("states*.npz"):
            z=np.load(q)
            print(str(q),[(x,z[x].shape) for x in z.files])
print("=== REINSTALL TRIAL FOR CENTERING DIAGNOSTIC ===")
T270=load_file("/work/checkpoint_trial264.safetensors")
for n in ("core.flow.a","core.a_in"):
    save_edit(n,T270[n])
print("private flow:",np.asarray(load("core.flow.a"))[[15,21,61,66]])
with open("/work/notes.md","a") as f:
    f.write("\n## Run 270: prepare factual-centering recalibration\n"
            "- Reinstalled only trial flow/a_in from checkpoint_trial264; still an unaccepted diagnostic.\n"
            "- Plan: fresh training GPU states, cross-validated ordinary-feature centering, then GPU factual and rollout acceptance.\n")
```