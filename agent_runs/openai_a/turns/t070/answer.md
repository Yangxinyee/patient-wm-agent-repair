The records support a substantial insulin effect, but the rejected decoder did not expose it in rollouts. I’ll check whether dose innovations are recoverable from earlier model features. Good held-out decoding there would support a more localized edit; poor decoding would argue against it.

```python
import inspect
import numpy as np
import torch
from pathlib import Path

print("FUNCTION INTERFACES")
for name in ["cpu_head_parts", "cpu_head_probs", "make_head_features",
             "repair_head_features", "head_dose_data", "head_unit_target",
             "make_policy_data", "policy_predictions_panel", "causal_shift_table"]:
    f = globals().get(name)
    if callable(f):
        print("\n", name, inspect.signature(f))
        if hasattr(f, "__code__"):
            print("names:", f.__code__.co_names)
            print("constants:", repr(f.__code__.co_consts)[:1300])

print("\nRETAINED ARRAYS")
for k in sorted(list(globals())):
    v = globals()[k]
    if isinstance(v, (np.ndarray, torch.Tensor)) and v.ndim and v.shape[0] >= 100:
        print(k, tuple(v.shape))

print("\nHEAD METADATA")
print(head_meta_train.head(3).to_string(index=False))
print(head_meta_extra.head(3).to_string(index=False))
print("\nCAUSAL SUMMARIES")
for name in ["base_rd", "insulin_rd", "rd_checks"]:
    if name in globals():
        print(name, repr(globals()[name])[:1800])

text = Path("/work/notes.md").read_text()
print("\nIDENTIFICATION NOTEBOOK, CONTINUED")
print(text[7200:10800])
```
