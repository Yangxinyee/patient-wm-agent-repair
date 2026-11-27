The recovered notes support an insulin effect near −1.9 per unit/hour, with persistence near 0.83; norepinephrine estimates remain less secure. The GPU response is too small for that insulin target. I’ll check whether the graft candidates scale linearly and recover the remaining identification evidence. Linear scaling would let me strengthen the edit without refitting it.

```python
import torch
from pathlib import Path

def brief_structure(obj):
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        return (type(obj).__name__, tuple(obj.shape))
    if isinstance(obj, dict):
        return {str(k): brief_structure(v) for k, v in list(obj.items())[:8]}
    if isinstance(obj, (list, tuple)):
        return [brief_structure(v) for v in obj[:8]]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return type(obj).__name__

print("ROBUST INPUTS", brief_structure(robust_X))
print("ROBUST PARTS", brief_structure(robust_parts))
print("ROBUST COSTS", brief_structure(robust_costs3))
print("GRAFT FUNCTION GLOBALS", make_insulin_graft.__code__.co_names)
print("PROBABILITY FUNCTION GLOBALS", graft_probs.__code__.co_names)

graft_zero = {n: torch.as_tensor(v).clone()
              for n, v in robust_candidates3[0.0].items()}
graft_unit = {
    n: (torch.as_tensor(robust_candidates3[-2.0][n]) - graft_zero[n]) / -2.0
    for n in graft_zero
}
print("\nAFFINE SCALING CHECK")
for n in graft_zero:
    current = torch.as_tensor(load(n))
    print(n, "unit norm", float(graft_unit[n].norm()),
          "current vs beta=-2", float(
              (current - torch.as_tensor(robust_candidates3[-2.0][n])).abs().max()))
    for btest in [-4.0, -8.0]:
        discrepancy = (torch.as_tensor(robust_candidates3[btest][n])
                       - (graft_zero[n] + btest * graft_unit[n])).abs().max()
        print(" ", btest, "affine error", float(discrepancy))

print("\nIDENTIFICATION NOTES, CONTINUED")
print(Path("/work/notes.md").read_text()[5600:11200])
```
