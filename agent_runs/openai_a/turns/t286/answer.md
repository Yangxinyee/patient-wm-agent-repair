The insulin threshold checks support a one-hour effect near −1.93 per unit and persistence near 0.83, with little evidence of an effect before treatment. I’ll now locate the exact decoder construction and saved diagnostics. Recovering its gating and residual terms will let me test the asymmetry without discarding the validated checkpoint.

```python
import inspect
import numpy as np
import torch

print("=== FUNCTIONS ===")
_fn = []
for k, v in list(globals().items()):
    if inspect.isfunction(v):
        _fn.append(k)
        print(k, inspect.signature(v))

print("\n=== EDIT AND CACHE VARIABLE NAMES ===")
print(", ".join(sorted(
    k for k in globals()
    if not k.startswith("__") and any(s in k.lower() for s in
        ["sc_", "select", "graft", "synth", "contrast", "candidate", "gate", "fit3"])
)))

print("\n=== CURRENT GATE CONFIGURATION ===")
for k, v in activity_gate_config.items():
    print(k, (("array", v.shape, "norm", float(np.linalg.norm(v)))
              if isinstance(v, np.ndarray) else v))

print("\n=== CHANGED HIDDEN ROWS / OUTPUT COLUMNS ===")
for suffix, axis in [("0.weight", 1), ("2.weight", 0)]:
    name = "attr_heads.value_bin." + suffix
    a = np.asarray(load(name))
    o = np.asarray(load_original(name))
    diff = np.linalg.norm(a-o, axis=axis)
    idx = np.flatnonzero(diff > 1e-7)
    print(name, "changed indices", idx.tolist(),
          "delta norms", np.round(diff[idx], 5).tolist())
print("hidden bias changes:",
      [(int(i), float(load("attr_heads.value_bin.0.bias")[i]),
        float(load_original("attr_heads.value_bin.0.bias")[i]))
       for i in np.flatnonzero(np.asarray(load("attr_heads.value_bin.0.bias")) !=
                              np.asarray(load_original("attr_heads.value_bin.0.bias")))])

print("\n=== NOREPINEPHRINE NOTE EXCERPTS ===")
_lines = _notes.splitlines()
for i, line in enumerate(_lines):
    if any(s in line.lower() for s in
           ["norepinephrine", "ne effect", "ne causal", "ne flexible", "norepi"]):
        if i > 140:
            print("\n".join(_lines[max(0, i-1):i+4])[:600])

print("\n=== BUILDER DETAILS ===")
for k in _fn:
    if any(s in k.lower() for s in ["candidate", "selective", "head_edit"]):
        f = globals()[k]
        try:
            print(k, inspect.getsource(f))
        except (OSError, TypeError):
            print(k, "names", f.__code__.co_names,
                  "locals", f.__code__.co_varnames,
                  "constants", repr(f.__code__.co_consts)[:1000])
```
