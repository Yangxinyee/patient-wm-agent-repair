The one-hour infusion records let us align readings with delivered doses. I’ll check which dose estimates, residuals, and head features are already available, then quantify the effect missing from the model’s predictions. A head edit is plausible only if its features retain information about unexpected doses.

```python
import numpy as np
import pandas as pd
import dis

def brief26(name):
    obj = globals().get(name)
    print(name, type(obj).__name__, getattr(obj, "shape", None))
    if isinstance(obj, dict):
        for key, val in list(obj.items())[:8]:
            print(" ", key, type(val).__name__, getattr(val, "shape", None))
    elif isinstance(obj, pd.DataFrame) and obj.size < 1000:
        print(obj.to_string(index=False))
    elif isinstance(obj, (list, tuple)):
        print(" length:", len(obj), "first:", obj[:8])

print("=== FEATURES AND RECORD ESTIMATES ===")
for nm26 in (
    "hidden23", "local_hidden", "hidden21", "dose_residuals",
    "outcome_residuals", "dose_models", "outcome_models", "DML",
    "dml_table", "rd_table", "basic_effects", "irf_coefficients"
):
    brief26(nm26)

print("\n=== NUMERIC NORMALIZATION ===")
print("numeric_note:", VOCAB.get("numeric_note"))
for key, val in VOCAB["numeric"].items():
    if isinstance(val, dict):
        print(key, "keys:", list(val))
        for subkey, subval in val.items():
            if not isinstance(subval, (list, dict)):
                print(" ", subkey, subval)
            elif isinstance(subval, dict):
                print(" ", subkey, {
                    k: v for k, v in subval.items()
                    if not isinstance(v, (dict, list))
                })

print("\n=== HELPER INTERFACES ===")
for nm26 in ("head_exact", "reconstruct_head", "make_panel", "clustered_ols"):
    fn26 = globals().get(nm26)
    if callable(fn26):
        print(nm26, "args:", fn26.__code__.co_varnames[:fn26.__code__.co_argcount],
              "defaults:", fn26.__defaults__)

print("\n=== HEAD_EXACT IMPLEMENTATION ===")
dis.dis(head_exact)

print("\n=== AVAILABLE STAY SUBSETS ===")
for nm26 in ("train_ids", "tune_ids", "tune_dev", "tune_check", "tune_holdout"):
    obj26 = globals().get(nm26)
    if obj26 is not None:
        arr26 = np.asarray(obj26)
        print(nm26, "shape:", arr26.shape, "first:", arr26.reshape(-1)[:12])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Python 26; forward 4\n"
        "- Checking retained dose innovations and value-head features before fitting a correction.\n"
        "- Proposed diagnostic: original prediction errors versus pre-dose-adjusted treatment innovations.\n"
        "- No weights changed.\n"
    )
```