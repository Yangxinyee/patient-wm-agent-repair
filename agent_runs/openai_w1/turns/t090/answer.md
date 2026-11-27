All bundle counts match, and END-only posteriors match exactly. Drug-only errors are small, while reading errors are much larger; that points toward a measurement-embedding error. I’ll inspect its categorical fields and recover the expected-dose fit layout. Fixing the measurement embedding would make latent-effect calibration substantially more reliable.

```python
import dis
import numpy as np
import torch
from pathlib import Path

print("=== ATOM EMBEDDING LOGIC, AFTER NUMERIC FIELDS ===")
_ins = list(dis.get_instructions(r36_atom))
for _i in _ins:
    if _i.offset >= 390:
        print(f"{_i.offset:4d} {_i.opname:24s} {_i.argrepr[:110]}")
print("=== CATEGORICAL TABLE SHAPES ===")
for _name in NAMES:
    if _name.startswith(("embed.categorical.", "embed.ekind", "embed.kind", "embed.kq", "embed.text")):
        print(_name, SHAPES[_name])
print("=== EXPECTED-DOSE FIT STRUCTURE ===")
for _drug, _fit in rr51_assignment.items():
    print(_drug, type(_fit).__name__, len(_fit))
    for _j, _v in enumerate(_fit):
        if isinstance(_v, (np.ndarray, torch.Tensor)):
            print(_j, type(_v).__name__, tuple(_v.shape),
                  "range", float(np.nanmin(np.asarray(_v))), float(np.nanmax(np.asarray(_v))))
        else:
            print(_j, str(_v)[:180])
print("=== CAUSAL SCALE / KERNEL NOTEBOOK LINES ===")
_text = Path("/work/notes.md").read_text()
_lines = [s for s in _text.splitlines()
          if any(t in s.lower() for t in ("kernel", "deconvol", "per unit", "mg/", "rate unit", "true shift"))]
for _line in _lines[:18]:
    print(_line[:350])
with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 80-81\nAll 8443 cached bundle counts match exported counts exactly. END-only z posteriors match exactly; drug-only median/max per-row RMSE 0.00277/0.04259; reading median 0.13633. END routing is correct. Suspect measurement categorical embedding fields, not bundle alignment. Run 80 stopped while inspecting tuple-valued expected-dose fits. No edits. Counts: 81 Python runs, 6 forward passes.\n")
```