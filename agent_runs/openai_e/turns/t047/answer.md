The retained data include the full hourly panels, original GPU states, and synthetic doubled/halved drug encodings. That is enough to attempt a reversible correction rather than rebuild everything. I’ll inspect the existing helpers and effect-fit diagnostics next. We need a feature that responds to changed doses but stays small on recorded doses; poor separation would rule out that approach.

```python
import inspect
import numpy as np
from pathlib import Path

print("=== HELPERS ===")
for key, obj in list(globals().items()):
    if callable(obj) and getattr(obj, "__module__", "") == "__main__" and key not in {
        "load", "load_original", "save_edit", "drop_edit", "edited"
    }:
        try:
            print(key, inspect.signature(obj))
        except Exception:
            print(key)
print("=== RELEVANT NOTES ===")
text43 = Path("/work/notes.md").read_text()
a43 = text43.find("## Run 10")
b43 = text43.find("## Run 21")
print(text43[a43:b43 if b43 > a43 else a43 + 2600][:2600])
print("=== CAUSAL FITS ===")
print("dynamic parameters:", getattr(dyn_fit, "x", None))
print(rd_table.to_string(index=False))
print("=== COUNTERFACTUAL FEATURE FITS ===")
for key, obj in cf_fits.items():
    print(key, type(obj).__name__,
          list(obj.keys()) if isinstance(obj, dict) else
          [getattr(v, "shape", None) for v in obj] if isinstance(obj, (list, tuple)) else getattr(obj, "shape", None))
print("=== RELEVANT TENSORS ===")
for name in NAMES:
    if any(s in name for s in (
        "z_post", "z_head", "z_in", "delta_in", "a_in", "c_out", "b_in",
        "attr_heads.value", "last_numeric", "transition.value"
    )):
        print(name, SHAPES[name])
print("=== UNITS AND KINDS ===")
print(RECON_UNITS, RECON_RATE_UNITS, RECON_EKINDS, END_CFG)
print("=== FORWARD FILES ===")
for folder in sorted(Path("/work").glob("fwd_*")):
    print(folder.name, [p.name for p in folder.iterdir()])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 43: targeted helper and effect diagnostics\n"
            "Retained panels, GPU states, and synthetic action contrasts are available. "
            "Inspecting exact helpers and effect identification before a reversible posterior-path edit. "
            "No edits. Python runs: 43; forward passes: 3.\n")
```
