The detector was fitted on all four slots, so missing slot coverage does not explain its failure. The remaining possibilities are recurrent amplification or an unsuitable MAP direction. I’ll inspect those scales and the prior decoder-calibration result; a new direction is justified only if it can reduce collateral changes without weakening NE sensitivity.

```python
from pathlib import Path
import numpy as np
import torch
import dis

ss204 = Path("/work/notes.md").read_text().split("\n## ")
print("=== PRIOR CALIBRATION RESULT ===")
for s in ss204:
    if s.splitlines()[0].startswith("Run 182:"):
        print(s[:1900])

print("\n=== EARLY CAUSAL ESTIMATION ===")
for s in ss204[:4]:
    print(s[:650])

print("\n=== SAVED EDIT SCALES ===")
for fn in ["baseline77.npz", "ne_candidate82.npz", "ne_trial103.npz",
           "ne_candidate151.npz", "ne_candidate199.npz"]:
    with np.load("/work/"+fn) as a:
        print(fn, {k: (float(a[k]) if a[k].ndim == 0
                       else {"shape": a[k].shape, "norm": round(float(np.linalg.norm(a[k])), 4)})
                   for k in a.files})

print("\n=== CURRENT CONTRAST DATA ===")
for k, v in C197.items():
    print(k, {s: getattr(a, "shape", type(a).__name__) for s, a in v.items()})
print("detector factual slot RMS:",
      float(np.sqrt(np.mean((np.asarray(F197) @ feature199)**2))))
print("direction199 norm:", float(np.linalg.norm(direction199)))
print("J199:", J199.shape)

print("\n=== DECODER INPUT/OUTPUT OBJECTS ===")
for name in sorted(list(globals())):
    obj = globals()[name]
    if name.endswith(("160", "162", "175", "176", "181", "182")):
        if hasattr(obj, "state_row") or hasattr(obj, "dt"):
            print(name, type(obj).__name__, str(getattr(obj, "columns", []))[:180])
        elif isinstance(obj, dict):
            print(name, "dict", {k: getattr(v, "shape", type(v).__name__)
                               for k, v in list(obj.items())[:5]})
        elif isinstance(obj, (np.ndarray, torch.Tensor)) and (
            "hidden" in name or "static" in name or "prob" in name or "centre" in name or "transition" in name):
            print(name, tuple(obj.shape))
print("decoder return instructions:")
instructions204 = list(dis.Bytecode(decode_states175))
for i in instructions204[-15:]:
    print(i.opname, str(i.argval)[:110])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 204: rule out missing slot guards\n"
            "Detector factual guards do cover all four slots. Recovering edit scales "
            "and prior decoder-calibration findings before changing direction or head. "
            "No weights changed; baseline77 installed. Python runs: 204; forwards: 23.\n")
```
