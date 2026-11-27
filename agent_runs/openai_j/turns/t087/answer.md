The record-based kernel fits suggest larger effects than the original model predicts: roughly −11 glucose units for doubled insulin and +4 MAP units for doubled norepinephrine. The insulin magnitude is less securely identified. I’ll inspect the cached state/readout probe and its bin conventions; a useful candidate must strengthen the appropriate response without exceeding the factual-fit tolerance.

```python
import numpy as np
import pandas as pd
from pathlib import Path

print("=== LATER RECONSTRUCTION NOTES ===")
print(Path("/work/notes.md").read_text()[10200:12400])

print("\n=== HELPER FUNCTIONS ===")
lines = []
for k, obj in list(globals().items()):
    if callable(obj) and getattr(obj, "__module__", "") == "__main__":
        c = getattr(obj, "__code__", None)
        if c is not None:
            lines.append(f"{k}{c.co_varnames[:c.co_argcount]}")
print("\n".join(lines)[:2700])

print("\n=== CORE / READOUT CACHE SHAPES ===")
lines = []
wanted = ("core_", "prelude", "recon", "jac", "contrast", "head_", "value_", "cf_", "meta")
exact = {"PA", "PN", "PF", "PT", "PO", "TA", "TF", "TPA", "HF", "HX",
         "centres0", "centres_t", "base", "feat0", "W_original", "W_new",
         "b_new", "core_check_table", "action_rows", "atom_vectors",
         "best_inputs", "readout_inputs", "grid", "dose_grid", "factual_meta"}
for k, obj in list(globals().items()):
    if k in exact or any(x in k.lower() for x in wanted):
        if isinstance(obj, np.ndarray):
            lines.append(f"{k}: {obj.shape} {obj.dtype}")
        elif isinstance(obj, pd.DataFrame):
            lines.append(f"{k}: DataFrame {obj.shape}; {list(obj.columns)}")
        elif isinstance(obj, dict):
            lines.append(f"{k}: dict {list(obj.keys())[:10]}")
        elif hasattr(obj, "shape"):
            lines.append(f"{k}: {type(obj).__name__} {tuple(obj.shape)}")
        elif isinstance(obj, (list, tuple)):
            lines.append(f"{k}: {type(obj).__name__} length {len(obj)}")
print("\n".join(lines)[:2800])

print("\n=== VALUE-HEAD SHAPES AND CENTRE CONVENTIONS ===")
for n in NAMES:
    if n.startswith("attr_heads.value_bin"):
        print(n, SHAPES[n])
for key in ["value_bin_centres", "value_bin_edges", "normalisation", "config"]:
    v = VOCAB.get(key)
    print(key, str(v)[:350])
print("ACTIVE EDITS:", edited())
```