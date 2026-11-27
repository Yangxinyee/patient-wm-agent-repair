The reconstruction stopped because the atom-owner mapping contains excluded bundles with negative indices. It also exposed a sampled array that had overwritten an earlier global. This is an implementation failure, not evidence about drug effects.

I’ll build an isolated reconstruction from the original weights and record bundles, rather than depend on mutable experimental globals. Exact posterior agreement will remain the required check before fitting an edit.

```python
import dis
import inspect
import numpy as np
import pandas as pd
import torch

print("=== EXPORTED STATE INDEX ===")
for r96_key, r96_value in repair_S.items():
    print(r96_key, type(r96_value).__name__, getattr(r96_value, "shape", ""))
print("selected index range:", int(np.min(r36_indices)), int(np.max(r36_indices)))
r96_export_reference = np.asarray(repair_S["x"])[r36_indices, -32:].copy()
print("reference shape:", r96_export_reference.shape)
print("selected stay/hour examples:",
      list(zip(np.asarray(repair_S["stay_id"])[r36_indices[:5]].tolist(),
               np.asarray(repair_S["hour"])[r36_indices[:5]].tolist())))
print("owner negative count:", int((r89_owner_all < 0).sum()))

print("\n=== STABLE RECORD OBJECTS ===")
for r96_name, r96_value in list(globals().items()):
    if isinstance(r96_value, pd.DataFrame) and len(r96_value) > 10000:
        print(r96_name, r96_value.shape, list(r96_value.columns)[:9])

print("\n=== BUNDLE CACHE INDEX ===")
for r96_name, r96_value in sorted(list(globals().items())):
    if r96_name.startswith(("r36_", "r71_")) and not callable(r96_value):
        if isinstance(r96_value, (np.ndarray, torch.Tensor, dict, list, tuple)):
            print(r96_name, type(r96_value).__name__,
                  tuple(r96_value.shape) if isinstance(r96_value, (np.ndarray, torch.Tensor)) else len(r96_value))

print("\n=== ATOM BUILDER: EMBEDDING ASSEMBLY ===")
for r96_instruction in dis.get_instructions(r36_atom):
    if r96_instruction.offset >= 400:
        print(r96_instruction.offset, r96_instruction.opname, r96_instruction.argrepr)

print("\n=== EMBEDDING TABLES ===")
for r96_name in NAMES:
    if r96_name.startswith("embed.") and not "transition" in r96_name:
        print(r96_name, SHAPES[r96_name])

with open("/work/notes.md", "a") as r96_f:
    r96_f.write(
        "\n## Runs 95–96\n"
        "Run 95 stopped on negative owner indices for excluded initial bundles; sampled globals also differ "
        "from earlier full-cache arrays. This is an implementation failure, not a model result. "
        "Building isolated reconstruction using original weights and stable records/state metadata. "
        "No tensors edited. Counts: 96 Python runs, 6 forward passes.\n"
    )
```
