The trial raises factual NLL by 0.01722, so it fails the constraint and is rejected. I’ll restore the insulin-only baseline and recover the earlier evidence and saved parameters. A smaller norepinephrine edit is worth testing only if it improves dose responses without breaking factual fit.

```python
from pathlib import Path
import numpy as np

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())

archive104 = np.load("/work/baseline77.npz")
print("BASELINE ARCHIVE", [(k, archive104[k].shape) for k in archive104.files])
b_original104 = load_original("core.b_in.weight")
if "delta" in archive104.files and archive104["delta"].shape == tuple(b_original104.shape):
    save_edit("core.b_in.weight", b_original104 + archive104["delta"])
    print("RESTORED baseline77 using saved delta")
else:
    print("RESTORE PENDING: inspect archive keys before interpreting weights")

notes104 = Path("/work/notes.md").read_text()
print("EARLY NOTEBOOK")
print(notes104[:16000])
print("RELEVANT KERNEL OBJECTS")
for key104, val104 in list(globals().items()):
    if key104.startswith("_") or key104.endswith("104"):
        continue
    if any(s in key104.lower() for s in ("baseline", "delta", "insulin", "target", "causal", "direction", "balanced", "beta", "nefit")):
        if isinstance(val104, np.ndarray):
            print(key104, "array", val104.shape)
        elif isinstance(val104, (dict, list, tuple)):
            print(key104, type(val104).__name__, "length", len(val104),
                  "keys", list(val104)[:12] if isinstance(val104, dict) else "")
        elif callable(val104):
            print(key104, "callable")
        elif isinstance(val104, (int, float, str)):
            print(key104, str(val104)[:180])
with Path("/work/notes.md").open("a") as f104:
    f104.write("\n## Run 104: reject corrected NE trial\n"
               "Forward13 factual original=1.60883, trial=1.62605; +.01722 violates .005. "
               "Trial103 rejected. Recovering baseline archive and prior causal evidence. "
               "Python runs: 104; forwards: 13.\n")
print("CURRENT EDITS", edited())
```