The rollout exactly repeats the weaker candidate’s four shifts. That suggests the stronger settings were explored but not actually saved. I’ll compare the staged tensors with the candidate matrices before making further judgments; this check will confirm which head is being evaluated.

```python
import numpy as np
import inspect
from pathlib import Path

T247 = np.vstack([
    load("attr_heads.value_bin.2.weight").numpy().T,
    load("attr_heads.value_bin.2.bias").numpy()[None, :]
]).astype(np.float64)

print("=== ACTUAL STAGED HEAD ===")
for name, candidate in [
    ("T235", T235),
    ("chosen235 T", chosen235["T"]),
    ("grid .3/.3", expanded234[("pair4", 0.1, 0.3, 0.3)]["T"]),
    ("grid .5/.4", grid242[(0.5, 0.4)]["T"]),
    ("grid .4/.6", grid242[(0.4, 0.6)]["T"])
]:
    arr = np.asarray(candidate)
    print(name, "RMS difference", float(np.sqrt(np.mean((arr-T247)**2))),
          "max difference", float(np.max(np.abs(arr-T247))))

print("=== HELPER SIGNATURES ===")
for name in ["score234", "centre234", "fit_arch215"]:
    fn = globals()[name]
    print(name, inspect.signature(fn), "globals:", fn.__code__.co_names,
          "constants:", str(fn.__code__.co_consts)[:500])

print("=== SAVED CANDIDATE DETAILS ===")
print("chosen235:", {k: compact240(v) for k, v in chosen235.items()
                     if k not in ["arch", "T", "Ftune"]})
print("st242 scale min/max:", float(np.min(st242["scale"])),
      float(np.max(st242["scale"])))
print("grid candidate norm:", float(np.linalg.norm(grid242[(0.5, 0.4)]["T"])))
print("actual head norm:", float(np.linalg.norm(T247)))

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 247 / restart tensor verification\n"
            "Independent 100-stay factual audit: delta +.00393. Forward18 reproduces "
            "[-7.878,+3.322,+2.231,-1.038] exactly. Exploratory grid settings must "
            "not be confused with saved tensors; comparing actual final-head weights "
            "against T235 and grid242 before staging any new candidate.\n")
```
