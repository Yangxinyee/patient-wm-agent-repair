The staged controller worsened value NLL by 0.137 nats—well beyond the allowed 0.005. I’ll undo that stage, retain the previously neutral attention reconstruction, and recover the implementation details. The next design must preserve ordinary state dynamics before adding any causal correction.

```python
from pathlib import Path
import numpy as np
import torch
import inspect

drop_edit("core.b_in.weight")
drop_edit("core.c_out.weight")
assert tuple(Wrepair125.shape) == tuple(SHAPES["core.prelude.0.attn.out.weight"])
save_edit("core.prelude.0.attn.out.weight", torch.as_tensor(Wrepair125, dtype=torch.float32))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 162 / staged controller rejected\n"
            "Forward 6 (fwd_7), 100 tuning stays: original value NLL 1.63164; "
            "controller+wire edited NLL 1.76833, delta +0.13669. "
            "Rejected before causal readout. Dropped b_in/c_out edits and restored "
            "the previously NLL-neutral reconstruction-only attention out Wrepair125; "
            "reconstruction kv retained. Need diagnose recurrent/off-manifold leakage.\n")

print("CURRENT EDITS", edited())
print("\n=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== EARLY NOTEBOOK / CAUSAL EVIDENCE ===")
print(Path("/work/notes.md").read_text()[:12000])
print("\n=== LOCAL HELPERS ===")
for name, obj in list(globals().items()):
    if inspect.isfunction(obj) and getattr(obj, "__module__", "") == "__main__":
        try:
            sig = str(inspect.signature(obj))
        except Exception:
            sig = "?"
        print(name, sig)
print("\n=== RECOVERED COMPONENTS ===")
for name in ["P125", "T125", "QE125", "SP125", "Vnew125", "Wrepair125",
             "Ccontroller145", "Cperp145", "Uexact145", "R118", "controllers159",
             "routes159", "wire_out152", "chosen125"]:
    obj = globals().get(name)
    if hasattr(obj, "shape"):
        a = np.asarray(obj)
        print(name, a.shape, "norm", float(np.linalg.norm(a)))
    elif isinstance(obj, dict):
        print(name, "dict keys", list(obj.keys()))
    else:
        print(name, type(obj).__name__, repr(obj)[:700])
print("\n=== STATE / CACHE VARIABLE NAMES ===")
print([n for n in globals() if any(s in n.lower() for s in
       ["cache", "state", "kernel", "pole", "lag", "readout", "weight", "route", "wire", "ctrl", "controller"])])
```