The latent projection weights are not unusually small, so their scale alone does not explain the pilot’s failure. The inspection stopped on a harmless namespace-printing error after restoring the original model. I’ll fix that inspection and recover the available state arrays and core layers. These will let us test whether the recurrent update suppresses the proposed latent direction.

```python
import builtins as bi
import types
from pathlib import Path

print("CURRENT EDITS:", edited())
print("\n=== RETAINED ARRAYS AND FIT DICTIONARIES ===")
r64_lines = []
for r64_key, r64_obj in bi.list(globals().items()):
    if r64_key.startswith("_") or r64_key.startswith("r64_"):
        continue
    r64_shape = getattr(r64_obj, "shape", None)
    if isinstance(r64_shape, (bi.tuple, torch.Size)):
        r64_lines.append(f"{r64_key}: {bi.type(r64_obj).__name__}{bi.tuple(r64_shape)}")
    elif isinstance(r64_obj, bi.dict) and bi.any(
        s in r64_key.lower() for s in ["fit", "caus", "effect", "dose", "repair", "weight", "state", "pilot"]
    ):
        r64_lines.append(f"{r64_key}: dict keys={bi.list(r64_obj)[:10]}")
print("\n".join(r64_lines)[:6800])

print("\n=== CORE TENSOR LAYOUT ===")
for r64_name in NAMES:
    if r64_name.startswith(("core.prelude.", "core.prelude_norm.", "core.blocks.",
                           "core.coda.", "core.out_norm.", "core.slots", "core.flow.", "core.age.")):
        print(r64_name, SHAPES[r64_name])

print("\n=== DATA SCHEMA ===")
print(Path("/data/README.md").read_text()[:2000])

with Path("/work/notes.md").open("a") as r64_file:
    r64_file.write("\nPython runs 63-64; forwards 6. All pilot edits reverted. "
                   "z_in has ordinary weight scale, so failure cannot be attributed simply "
                   "to a tiny latent projection. Recovering retained state data and exact core layout.\n")
```
