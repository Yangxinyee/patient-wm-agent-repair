The helper does exclude drug rows, but the output was truncated before we could establish whether END atoms were actually present in the tested bundles. I’ll compare the reconstructed encoding with each cached input block and inspect the fit parameters compactly. A close match to one block would indicate an indexing issue; uniformly poor matches would point back to encoding.

```python
import numpy as np
import torch

print("=== ENCODING ALIGNMENT ===")
_e63 = rr58_e0.detach().cpu().numpy()
for _j63 in range(3):
    _b63 = rr48_input0[rr58_mi, 192*_j63:192*(_j63+1)]
    print("input block", _j63,
          "RMSE", round(float(np.sqrt(np.mean((_e63-_b63)**2))), 6),
          "cosine", round(float(np.sum(_e63*_b63) /
                              np.sqrt(np.sum(_e63**2)*np.sum(_b63**2))), 6))
print("sample row fields:", rr58_mrows[0][0]._fields)
print("sample bundle:", [(r.event, r.hour) for r in rr58_mrows[0]])
print("presence counts:", rr48_present.sum(axis=0))
print("rates range:", rr48_rates.min(axis=0), rr48_rates.max(axis=0))
print("assignment input agreement:",
      float(np.sqrt(np.mean((rr51_assign_X-rr48_input0[:, :384])**2))))

print("\n=== EXPECTED-DOSE FIT LAYOUT ===")
for _drug63, _fit63 in rr51_assignment.items():
    print(_drug63, "tuple length", len(_fit63))
    for _j63, _part63 in enumerate(_fit63):
        if isinstance(_part63, (np.ndarray, torch.Tensor)):
            _a63 = np.asarray(_part63)
            print(" ", _j63, tuple(_a63.shape),
                  "min/max", np.round([_a63.min(), _a63.max()], 5))
        else:
            print(" ", _j63, repr(_part63)[:120])

print("\n=== RETAINED ALIGNMENT OBJECTS ===")
_count63 = 0
for _nm63 in sorted(list(globals())):
    if _nm63.startswith(("r34", "r35", "r36", "r42", "r43", "r48")):
        _v63 = globals()[_nm63]
        if isinstance(_v63, (np.ndarray, torch.Tensor)):
            print(_nm63, tuple(_v63.shape))
        elif isinstance(_v63, dict):
            print(_nm63, "dict", list(_v63)[:4])
        elif isinstance(_v63, (list, tuple)):
            print(_nm63, type(_v63).__name__, len(_v63))
        elif callable(_v63):
            print(_nm63, "callable", _v63.__code__.co_varnames[:4]
                  if hasattr(_v63, "__code__") else "")
        else:
            continue
        _count63 += 1
        if _count63 >= 45:
            break
print("CURRENT EDITS:", edited())
with open("/work/notes.md", "a") as _f63:
    _f63.write("\n## Run 63\n"
               "Run 62 disassembly exceeded output limit. Checking encoding/input alignment "
               "and expected-dose tuple layout with bounded output. No weights changed. "
               "Counts: 63 Python runs, 6 forward passes.\n")
```