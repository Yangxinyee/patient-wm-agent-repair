This is still the Run 236 output, so no new result or weight change is confirmed. I’ll keep the next extraction short enough to avoid truncation. Numerical kernels and the current response decomposition are needed to distinguish a justified calibration from arbitrary rollout tuning.

```python
import numpy as np
import pandas as pd
import torch
import inspect

def short237(v, depth=0):
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    if isinstance(v, np.ndarray):
        return np.round(v, 5).tolist() if v.size <= 24 else {"shape": tuple(v.shape)}
    if isinstance(v, pd.DataFrame):
        return {"shape": v.shape, "columns": list(v.columns),
                "head": v.head(2).to_dict("records")}
    if isinstance(v, dict):
        return {str(k): short237(x, depth + 1) for k, x in list(v.items())[:7]} if depth < 2 else list(v)[:7]
    if isinstance(v, (list, tuple)):
        return [short237(x, depth + 1) for x in v[:6]] if depth < 2 else {"length": len(v)}
    if isinstance(v, (str, int, float, np.number)):
        return str(v)[:120]
    return type(v).__name__

print("=== RECORD-DERIVED KERNELS ===")
for name in ["causal_beta80", "kernels137", "provisional_kernels", "target_table89"]:
    if name in globals():
        print(name, str(short237(globals()[name]))[:1100])

print("=== CURRENT CHANNEL DECOMPOSITION ===")
for name in ["dual_candidate193", "dual_b202", "connection_weight123",
             "connection_decoder123", "connection_direction123",
             "connection_scale123", "coef_ne186", "gate_ne186",
             "gate_bias186", "gate_scale94", "gate_bias94", "coef209"]:
    if name in globals():
        print(name, str(short237(globals()[name]))[:220])

g237 = dict(globals())
print("=== CONSTRUCTION VARIABLE NAMES ===")
for suffix in ["193", "194", "202", "206", "209", "232", "233"]:
    print(suffix, ", ".join(sorted(k for k in g237 if k.endswith(suffix)))[:350])

print("=== EVALUATION DEPENDENCIES ===")
for name in ["factual_eval194", "collect_channel186", "collect_selected187",
             "gated_ne185", "replay_factual125"]:
    obj = g237.get(name)
    if inspect.isfunction(obj):
        print(name, str(obj.__code__.co_names)[:450])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 237\nCompact numerical kernel and repair-interface extraction; no saved weights changed.\n")
```
