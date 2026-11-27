The state layout is correct to numerical precision, so state alignment is not the source of the error. The remaining suspects are the register inputs and the distinction between prior context and prediction view. I’ll inspect those calculations next; separating their gap terms would be justified only if it improves agreement across sensors, not just average glucose.

```python
import dis
from collections import defaultdict
import numpy as np

print("=== COMPACT HEAD HELPER, REGISTER AND LATENT SECTION ===")
_groups = defaultdict(list)
_line = None
for _ins in dis.get_instructions(reconstruct_head):
    if _ins.starts_line is not None:
        _line = _ins.starts_line
    if _line is not None and _line >= 94:
        _arg = _ins.argrepr.replace("NULL|self + ", "").replace("NULL + ", "")
        _groups[_line].append(_ins.opname + (":" + _arg if _arg else ""))
for _line, _ops in _groups.items():
    print(str(_line) + " " + " | ".join(_ops)[:650])

print("\n=== HELPER CONSTANTS ===")
print(reconstruct_head.__code__.co_consts)

print("\n=== ERRORS BY REGISTER AVAILABILITY ===")
for _event in MEAS:
    for _has in (False, True):
        _m = (head_rows.event == _event) & (head_rows.last_value.notna() == _has)
        _r = head_rows.loc[_m]
        if len(_r):
            _d = _r.local_mean.to_numpy() - _r.predicted_mean.to_numpy()
            print(_event, "has_last", _has, "n", len(_r),
                  "bias", round(float(np.mean(_d)), 4),
                  "RMS", round(float(np.sqrt(np.mean(_d**2))), 4))

print("\n=== STATE AND FACTUAL HOURS ===")
print("state hours:", np.unique(state_hours)[:18])
print("reference hours:", np.unique(head_reference.hour)[:18])
print("head weight names:", head_weight_names)
```
