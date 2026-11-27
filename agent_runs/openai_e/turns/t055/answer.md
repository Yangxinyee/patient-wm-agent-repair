The edit strengthens insulin responses: doubling now predicts −6.81 rather than −1.67, and halving predicts +2.54 rather than +0.78. It leaves norepinephrine largely unchanged. I’ll inspect the fitted correction and factual errors to find a more selective version; it must retain the rollout gain while bringing NLL within the limit.

```python
import numpy as np
import torch
import inspect
from pathlib import Path
import pandas as pd

trial_b_weight = load("core.b_in.weight")
trial_b_weight = trial_b_weight.clone() if torch.is_tensor(trial_b_weight) else trial_b_weight.copy()
trial_b_delta = trial_b_weight - load_original("core.b_in.weight")

print("NOTEBOOK, CAUSAL/EDIT DETAILS")
recovery_notes = Path("/work/notes.md").read_text()
print(recovery_notes[5500:12000])

print("\nFUNCTIONS")
for recovery_name, recovery_obj in list(globals().items()):
    if callable(recovery_obj) and getattr(recovery_obj, "__module__", "") == "__main__":
        try:
            print(recovery_name, inspect.signature(recovery_obj))
        except Exception:
            print(recovery_name)

print("\nDATAFRAMES")
for recovery_name, recovery_obj in list(globals().items()):
    if isinstance(recovery_obj, pd.DataFrame) and not recovery_name.startswith("recovery"):
        print(recovery_name, recovery_obj.shape, recovery_obj.columns.tolist())

print("\nRELEVANT ARRAYS/SCALARS")
for recovery_name, recovery_obj in list(globals().items()):
    if recovery_name.startswith("recovery") or recovery_name.startswith("_"):
        continue
    if any(s in recovery_name.lower() for s in
           ["prel", "contrast", "sens", "direction", "coef", "anchor", "beta", "rho", "target", "fit", "probe", "strength"]):
        if isinstance(recovery_obj, (np.ndarray, torch.Tensor)):
            print(recovery_name, tuple(recovery_obj.shape))
        elif isinstance(recovery_obj, (int, float, str)):
            print(recovery_name, str(recovery_obj)[:160])
        elif isinstance(recovery_obj, dict):
            print(recovery_name, "keys", list(recovery_obj)[:15])

factual4_original = pd.read_csv("/work/fwd_4/factual_original.csv")
factual4_trial = pd.read_csv("/work/fwd_4/factual_edited.csv")
print("\nFACTUAL FILE COLUMNS", factual4_original.columns.tolist())
print(factual4_original.head(2).to_string(index=False))
print("TRIAL WEIGHT CHANGE NORM", float(torch.linalg.vector_norm(torch.as_tensor(trial_b_delta))))

with open("/work/notes.md", "a") as recovery_log:
    recovery_log.write(
        "\n## Forward 4–5 and Run 49: first trial validation\n"
        "On 50 fixed tuning stays, factual NLL original=1.64063, trial=1.64660 "
        "(+0.00597: fails limit). Paired rollouts original/trial: insulin x2 "
        "-1.672/-6.811; insulin half +0.784/+2.535; NE x2 -0.548/-0.453; "
        "NE half -0.726/-0.831. Insulin direction works but requires better "
        "selectivity or factual repair; trial not accepted. Saved trial weights "
        "in kernel for reversible comparisons. Python runs: 49; forwards: 5.\n"
    )
```