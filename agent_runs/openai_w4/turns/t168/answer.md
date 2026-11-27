On valid contexts, doubling insulin produces about 94% of the intended channel change. Compression cannot explain a roughly sevenfold shortfall in the sustained response. I’ll investigate the downstream response direction instead, aiming to match its time course without the recurrent amplification that made the earlier larger edit fail factual fit.

```python
import inspect
import numpy as np
import torch
import pandas as pd
from pathlib import Path

print("=== DYNAMIC INTERFACES ===")
for name149 in [
    "pre_context_repair", "transition_with_b110", "head_features_fixed",
    "head_predict_repair", "head_probs_repair", "capture_ep112"
]:
    fn149 = globals().get(name149)
    if callable(fn149) and hasattr(fn149, "__code__"):
        print(name149, inspect.signature(fn149))
        print("locals:", fn149.__code__.co_varnames)
        print("dependencies:", fn149.__code__.co_names)
        print("constants:", [
            c for c in fn149.__code__.co_consts
            if isinstance(c, (str, int, float, tuple))
        ][:22])

valid149 = np.flatnonzero(np.isfinite(reference_x125).all(axis=1))
idx149 = valid149[valid149 > 2][:3]
parts149 = pre_context_repair(repair_bank, idx149)
print("\npre_context return:", {
    k: (tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
    for k, v in parts149.items()
})
print("transition output:", type(transition73(idx149)).__name__,
      getattr(transition73(idx149), "shape", None))

print("\n=== READING FIXTURES ===")
for name149 in ["probe_indices95", "probe_data95", "head_train_data_repair"]:
    obj149 = globals()[name149]
    print(name149, getattr(obj149, "shape", None))
    if isinstance(obj149, pd.DataFrame):
        print(obj149.head(2).to_string(index=False))
print("response direction norms:",
      {k: round(float(np.linalg.norm(v)), 5) for k, v in directions118.items()})

print("\n=== AVAILABLE GPU ROLLOUT COMPARISONS ===")
for dir149 in sorted(Path("/work").glob("fwd_*"),
                     key=lambda p: int(p.name.split("_")[-1])):
    for file149 in sorted(dir149.glob("dose_edit_*.csv")):
        data149 = pd.read_csv(file149)
        if {"edit", "shift", "reached", "hour"}.issubset(data149.columns):
            keep149 = data149["reached"].astype(bool) & data149["hour"].between(13, 24)
            print(str(file149),
                  data149.loc[keep149].groupby("edit")["shift"].mean().round(4).to_dict())

with open("/work/notes.md", "a") as f149:
    f149.write(
        "\n## Run 149\n"
        "Insulin dose doubling channel slope is approximately 0.94, refuting compression "
        "as the main explanation for the sustained-effect shortfall. "
        "Recovering dynamic-response and reading interfaces for a constrained connection. "
        "No weights changed.\n"
    )
```