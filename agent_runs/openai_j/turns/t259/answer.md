The current MAP correction is a gated linear logit tilt. Its weak doubling response may reflect nonlinear dose information in the hidden state, but that is not yet established. I’ll inspect the cached action contrasts and neuron allocation without changing weights. They should reveal whether a selective doubling correction is feasible within the factual budget.

```python
import inspect
import types
import numpy as np
import torch

audit229_functions = [
    repair_core_rows, repair_changed_action, repair_core_step,
    wm_meta145, wm_mc145, wm_mc_raw146, trial158_audit,
    screen_map218, raw_coef208, head177_eval
]
audit229_fields = set()
for audit229_fn in audit229_functions:
    audit229_strings = [
        c for c in audit229_fn.__code__.co_consts if isinstance(c, str)
    ]
    audit229_fields.update(audit229_strings)
    print("\nFUNCTION", audit229_fn.__name__, inspect.signature(audit229_fn))
    print("globals:", audit229_fn.__code__.co_names)
    print("fields:", audit229_strings[:25])

print("\nRELEVANT CACHES")
audit229_seen = 0
for audit229_name, audit229_obj in list(globals().items()):
    if (isinstance(audit229_obj, dict)
        and audit229_name.startswith(("wm_", "repair_", "map", "trial"))
        and set(audit229_obj).intersection(audit229_fields)):
        print(audit229_name, {
            k: (type(v).__name__, tuple(v.shape) if hasattr(v, "shape") else "")
            for k, v in list(audit229_obj.items())[:10]
        })
        audit229_seen += 1
        if audit229_seen >= 20:
            break

print("\nINSULIN CONSTRUCTION VARIABLES")
for audit229_name in sorted(list(globals())):
    if audit229_name.startswith("trial158_"):
        audit229_obj = globals()[audit229_name]
        print(audit229_name, type(audit229_obj).__name__,
              getattr(audit229_obj, "shape", ""),
              audit229_obj if np.isscalar(audit229_obj) else "")

audit229_w0 = head177_base["attr_heads.value_bin.0.weight"]
audit229_d0 = audit229_w0 - torch.as_tensor(
    load_original("attr_heads.value_bin.0.weight")
)
audit229_changed = torch.argsort(
    torch.linalg.vector_norm(audit229_d0, dim=1), descending=True
)[:8]
print("\nMOST CHANGED INSULIN-BASE INPUT NEURONS:",
      [(int(j), float(torch.linalg.vector_norm(audit229_d0[j])))
       for j in audit229_changed])
print("TILT VECTOR:", wm_v151)
```
