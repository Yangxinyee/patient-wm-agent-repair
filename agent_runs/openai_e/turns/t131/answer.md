The current direction was fitted using only 31 next-reading contexts, and its detector has substantial factual activation relative to its dose contrast. That may explain why scaling it damages fit. I’ll measure the actual slot-level perturbations and recover the dose-innovation data. If factual activation tracks genuine dose innovations, an innovation-targeted correction may be safer than forcing the detector to zero.

```python
import inspect
from pathlib import Path
import numpy as np
import pandas as pd
import torch

with np.load("/work/ne_trial103.npz") as a:
    feature113 = a["feature"].astype(np.float64)
    direction113 = a["direction"].astype(np.float64)
    strength113 = float(a["strength"])

E113 = E_fact95.detach().cpu().numpy()
scores113 = E113 @ feature113
delta_step113 = torch.nn.functional.softplus(W0["core.delta_in"]).numpy()
perturb113 = scores113[:, :, None] * (
    strength113 * delta_step113 * direction113
)[None, None, :]

print("=== ACTUAL NE SLOT PERTURBATIONS, FULL TRIAL ===")
print("score slot means:", scores113.mean(axis=0))
print("score slot SD:", scores113.std(axis=0))
print("b_in update RMS per bundle:",
      np.quantile(np.sqrt(np.mean(perturb113**2, axis=(1, 2))),
                  [0, .25, .5, .75, .9, .99, 1]))
print("direction norm:", np.linalg.norm(direction113))
print("weighted direction norm:", np.linalg.norm(delta_step113 * direction113))
print("state columns:", list(state_df.columns))
print(state_df.head(4).to_string(index=False))
for col in state_df.columns:
    if any(s in col.lower() for s in ["insulin", "norepinephrine", "had_", "has_"]):
        values = state_df[col].to_numpy()
        for level in pd.unique(values):
            mask = values == level
            if mask.any():
                print(col, level, "n", int(mask.sum()),
                      "mean score", float(scores113[mask].mean()),
                      "score RMS", float(np.sqrt(np.mean(scores113[mask]**2))))

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

print("\n=== HEAD HELPER DETAILS ===")
for name in ["linear0", "activation0", "dose_covariates"]:
    fn = globals().get(name)
    if inspect.isfunction(fn):
        print(name, fn.__code__.co_names, repr(fn.__code__.co_consts)[:650])

print("\n=== CORRECTED DETECTOR OBJECTS ===")
for name, value in list(globals().items()):
    if name.endswith("103") and not callable(value):
        if isinstance(value, (np.ndarray, torch.Tensor)):
            print(name, tuple(value.shape))
        elif isinstance(value, (dict, list, tuple, int, float, str)):
            print(name, repr(value)[:350])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 113: actual perturbation and innovation audit\n"
            "Existing MAP direction used only 31 next-reading contexts. "
            "Measuring factual per-slot NE perturbations and recovering residual-dose "
            "targets to distinguish avoidable leakage from useful innovation signal. "
            "No weights changed. Python runs: 113; forwards: 15.\n")
```