The comparison stopped because one cached array is a PyTorch tensor and the other is NumPy. That failure tells us nothing about encoding accuracy, and no weights changed. I’ll normalize the types and rerun the same checks; the result should distinguish indexing from reconstruction error.

```python
import numpy as np
import torch

def rr64_np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

print("=== ENCODING ALIGNMENT ===")
_e64 = rr64_np(rr58_e0)
_x64 = rr64_np(rr48_input0)
_i64 = rr64_np(rr58_mi).astype(int)
for _j64 in range(3):
    _b64 = _x64[_i64, 192*_j64:192*(_j64+1)]
    print("input block", _j64,
          "RMSE", round(float(np.sqrt(np.mean((_e64-_b64)**2))), 6),
          "cosine", round(float(np.sum(_e64*_b64) /
                              np.sqrt(np.sum(_e64**2)*np.sum(_b64**2))), 6))
print("sample row fields:", rr58_mrows[0][0]._fields)
print("sample bundle:", [(r.event, r.hour) for r in rr58_mrows[0]])
print("presence counts:", rr64_np(rr48_present).sum(axis=0))
print("rates range:", rr64_np(rr48_rates).min(axis=0),
      rr64_np(rr48_rates).max(axis=0))
print("assignment input agreement:",
      float(np.sqrt(np.mean((rr64_np(rr51_assign_X)-_x64[:, :384])**2))))

print("\n=== EXPECTED-DOSE FIT LAYOUT ===")
for _drug64, _fit64 in rr51_assignment.items():
    print(_drug64, "tuple length", len(_fit64))
    for _j64, _part64 in enumerate(_fit64):
        if isinstance(_part64, (np.ndarray, torch.Tensor)):
            _a64 = rr64_np(_part64)
            print(" ", _j64, tuple(_a64.shape),
                  "min/max", np.round([_a64.min(), _a64.max()], 5))
        else:
            print(" ", _j64, repr(_part64)[:120])

print("\n=== RETAINED ALIGNMENT OBJECTS ===")
_count64 = 0
for _nm64 in sorted(list(globals())):
    if _nm64.startswith(("r36", "r42", "r43", "r48")):
        _v64 = globals()[_nm64]
        if isinstance(_v64, (np.ndarray, torch.Tensor)):
            print(_nm64, type(_v64).__name__, tuple(_v64.shape))
        elif isinstance(_v64, dict):
            print(_nm64, "dict", list(_v64)[:4])
        elif isinstance(_v64, (list, tuple)):
            print(_nm64, type(_v64).__name__, len(_v64))
        elif callable(_v64):
            print(_nm64, "callable")
        else:
            continue
        _count64 += 1
        if _count64 >= 50:
            break
print("CURRENT EDITS:", edited())
with open("/work/notes.md", "a") as _f64:
    _f64.write("\n## Runs 63-64\n"
               "Run 63 stopped on NumPy/Torch type mismatch. Run 64 converts cached arrays "
               "before checking alignment and fit layout. No weights changed. "
               "Counts: 64 Python runs, 6 forward passes.\n")
```