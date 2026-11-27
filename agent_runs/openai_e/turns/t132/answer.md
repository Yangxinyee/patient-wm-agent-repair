The audit stopped because the retained prelude features are NumPy arrays, not tensors. It produced no measurements and changed no weights. I’ll rerun it with explicit type handling; the test remains whether factual detector activation represents dose innovation or unwanted leakage.

```python
import inspect
from pathlib import Path
import numpy as np
import pandas as pd
import torch

def as_numpy114(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

with np.load("/work/ne_trial103.npz") as a:
    feature114 = a["feature"].astype(np.float64)
    direction114 = a["direction"].astype(np.float64)
    strength114 = float(a["strength"])

E114 = as_numpy114(E_fact95)
scores114 = E114 @ feature114
delta_step114 = torch.nn.functional.softplus(
    torch.as_tensor(W0["core.delta_in"])
).numpy()
perturb114 = scores114[:, :, None] * (
    strength114 * delta_step114 * direction114
)[None, None, :]

print("=== FULL-TRIAL NE SLOT PERTURBATIONS ===")
print("score slot means:", scores114.mean(axis=0))
print("score slot SD:", scores114.std(axis=0))
print("update RMS quantiles:",
      np.quantile(np.sqrt(np.mean(perturb114**2, axis=(1, 2))),
                  [0, .25, .5, .75, .9, .99, 1]))
print("direction / weighted direction norms:",
      np.linalg.norm(direction114),
      np.linalg.norm(delta_step114 * direction114))
print("state columns:", list(state_df.columns))
print(state_df.head(4).to_string(index=False))
for col in state_df.columns:
    if any(s in col.lower() for s in ["insulin", "norepinephrine", "had_", "has_"]):
        values = state_df[col].to_numpy()
        for level in pd.unique(values):
            mask = values == level
            if mask.any():
                print(col, level, "n", int(mask.sum()),
                      "mean", float(scores114[mask].mean()),
                      "RMS", float(np.sqrt(np.mean(scores114[mask]**2))))

print("\n=== DATAFRAME KEYS ===")
for name, value in list(globals().items()):
    if isinstance(value, pd.DataFrame) and not name.startswith("_"):
        if any(s in name.lower() for s in
               ["panel", "tune", "meta", "resid", "policy", "dose", "drug", "cf"]):
            print(name, value.shape, list(value.columns)[:12])

print("\n=== FEATURE / PREDICTION HELPERS ===")
for name, fn in list(globals().items()):
    if inspect.isfunction(fn) and any(
        s in name.lower() for s in ["prelude", "feature", "view", "predict", "mean",
                                  "dose_cov", "linear0", "activation0"]
    ):
        try:
            print(name, str(inspect.signature(fn))[:150])
        except (ValueError, TypeError):
            print(name)

print("\n=== HEAD / POLICY HELPER DETAILS ===")
for name in ["linear0", "activation0", "dose_covariates"]:
    fn = globals().get(name)
    if inspect.isfunction(fn):
        print(name, fn.__code__.co_names, repr(fn.__code__.co_consts)[:650])

print("\n=== RUN103 OBJECTS ===")
for name, value in list(globals().items()):
    if name.endswith("103") and not callable(value):
        if isinstance(value, (np.ndarray, torch.Tensor)):
            print(name, tuple(value.shape))
        elif isinstance(value, (dict, list, tuple, int, float, str)):
            print(name, repr(value)[:350])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Runs 113–114: robust perturbation audit\n"
            "Run113 stopped because E_fact95 is an ndarray; it changed no weights. "
            "Run114 repeats the diagnostic with explicit array/tensor conversion. "
            "Recovering factual dose-innovation targets before choosing a new detector. "
            "Python runs: 114; forwards: 15.\n")
```