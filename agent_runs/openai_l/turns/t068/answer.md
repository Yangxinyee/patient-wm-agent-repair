The encoder now matches the exported posterior states closely. Before using it to construct an edit, I’ll recover the transition equations and earlier causal evidence. A successful transition reconstruction would let us test corrections without mistaking implementation errors for drug effects.

```python
from pathlib import Path
import numpy as np
import torch

arch61 = Path("/model/ARCHITECTURE.md").read_text()
print("=== ARCHITECTURE ===")
print(arch61)

notes61 = Path("/work/notes.md").read_text()
print("\n=== EARLY CAUSAL EVIDENCE ===")
print(notes61[:6500])

print("\n=== RELEVANT RETAINED OBJECTS ===")
for n61, v61 in sorted(list(globals().items())):
    if n61.startswith("_"):
        continue
    if callable(v61) and any(s in n61.lower() for s in
                            ("enc", "bundle", "core", "trans", "r30", "r32", "r49", "r60", "posterior")):
        co61 = getattr(v61, "__code__", None)
        print(n61, "function",
              tuple(co61.co_varnames[:co61.co_argcount]) if co61 else "")
    elif any(s in n61.lower() for s in
             ("valid60", "state", "ctx", "kernels", "train_ids", "bundle49", "raw49")):
        if isinstance(v61, (np.ndarray, torch.Tensor)):
            print(n61, type(v61).__name__, tuple(v61.shape))
        elif isinstance(v61, dict):
            print(n61, "dict", list(v61.keys())[:12])
        elif isinstance(v61, (list, tuple)):
            print(n61, type(v61).__name__, "length", len(v61))

with open("/work/notes.md", "a") as f61:
    f61.write("\n## Run 61 / restart recovery\n"
              "Posterior reconstruction is validated to 3.55e-5 RMS. "
              "Recovering recurrence equations and original causal evidence before choosing "
              "a factual-preserving residual-dose weight edit. No tensors edited.\n")
```