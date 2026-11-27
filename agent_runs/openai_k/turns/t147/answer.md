The architecture confirms that dose changes enter through action embeddings and the posterior, while predictions also use a prior latent draw. That distinction may explain why the previous token edit preserved factual fit but worsened dose responses. I’ll inspect only the retained calibration functions and tensors next; they must distinguish the drugs and track dose changes before another rollout test.

```python
import re, inspect
import numpy as np, torch
from pathlib import Path

print("=== TARGETED FUNCTIONS ===")
for k, v in list(globals().items()):
    if inspect.isfunction(v) and re.search(
        r"(prior_features74|candidate|latent|token|recurrent|rollout|step|sequence|states|dose_decode)", k
    ):
        print(k, inspect.signature(v))
        if k == "prior_features74":
            print(" references:", v.__code__.co_names)
            print(" locals:", v.__code__.co_varnames)
            try:
                print(inspect.getsource(v))
            except (OSError, TypeError):
                pass

print("\n=== KEY ARRAYS / RECIPES ===")
for k in [
    "candidates90", "candidate88", "candidate90", "dr66", "delta88",
    "X66", "X74", "Y74", "B74", "B88", "U88", "V88",
    "state_keys_audit", "states_audit", "W0", "targets132"
]:
    if k not in globals():
        continue
    v = globals()[k]
    if isinstance(v, (np.ndarray, torch.Tensor)):
        print(k, tuple(v.shape), str(v.dtype))
    elif isinstance(v, dict):
        keys = list(v.keys())
        print(k, "keys:", keys[:15], "total:", len(keys))
        if k == "candidates90":
            for a, b in list(v.items())[:4]:
                print(" candidate", a, type(b).__name__,
                      list(b.keys()) if isinstance(b, dict) else "")
    else:
        print(k, type(v).__name__, "length:", len(v) if hasattr(v, "__len__") else "")

print("\n=== OTHER RELEVANT RETAINED NAMES ===")
names136 = [
    k for k, v in list(globals().items())
    if re.search(r"(66|74|88|90)$", k) and not inspect.isfunction(v)
]
print(names136)

print("\n=== EARLY IDENTIFICATION NOTES ===")
notes136 = Path("/work/notes.md").read_text()
print(notes136[:5200])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 136: targeted latent/recurrent interface recovery\n"
            "- Architecture separates posterior token effects from prior-head predictions.\n"
            "- Inspecting retained calibration recipes and early identification evidence.\n"
            "- No edits. Python runs: 136; forward passes: 6.\n")
```