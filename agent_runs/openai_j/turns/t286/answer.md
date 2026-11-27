The last 100 stays pass with a +0.00342-nat factual penalty, giving this edit a better margin than on the preceding batch. Norepinephrine doubling remains the main unresolved weakness. I’ll inspect the saved nonlinear MAP candidates before changing anything; one must improve doubling sensitivity on held-out states without consuming the factual margin.

```python
import inspect
import dis
import numpy as np
import pandas as pd
from pathlib import Path

def compact_source251(fn, limit):
    try:
        text = inspect.getsource(fn)
    except Exception:
        text = dis.Bytecode(fn).dis()
    return text[:limit]

print("=== EXACT EDIT CONSTRUCTION ===")
print(compact_source251(head177_make, 3400))
print("\n=== EXACT HEAD EVALUATION ===")
print(compact_source251(head177_eval, 1500))

print("\n=== NONLINEAR CANDIDATE HELPER ===")
print("dependencies:", curve_eval236.__code__.co_names)
print(compact_source251(curve_eval236, 1700))

print("\n=== RELEVANT CACHED OBJECTS ===")
for k, v in sorted(list(globals().items())):
    if k.startswith("_"):
        continue
    if (any(t in k for t in ["236", "237", "242", "247"])
        or k in head177_eval.__code__.co_names):
        if isinstance(v, np.ndarray):
            print(k, "array", v.shape)
        elif hasattr(v, "shape") and not isinstance(v, pd.DataFrame):
            print(k, type(v).__name__, tuple(v.shape))
        elif isinstance(v, dict):
            print(k, "keys", list(v)[:10])
        elif isinstance(v, pd.DataFrame):
            print(k, "DataFrame", v.shape, list(v.columns))
        elif isinstance(v, (int, float, str, list, tuple)):
            print(k, repr(v)[:130])

print("\n=== NONLINEAR FRONTIER ===")
df251 = pd.read_csv("/work/map_nonlinear_frontier.csv")
print(df251.head(6).to_string(index=False))
print("\n=== PRIOR GLUCOSE CALIBRATION SCREEN ===")
print(pd.read_csv("/work/glucose_calibration_frontier.csv").head(5).to_string(index=False))

with Path("/work/notes.md").open("a") as fh251:
    fh251.write(
        "\n## Run 251: independent factual pass and nonlinear review\n"
        "- Forward 29 / fwd_30: current safer edit value dNLL +0.00342"
        " on tuning stays 300–399.\n"
        "- Current edit has passed both second-half tuning batches; first-half"
        " checks and current rollout still needed.\n"
        "- No weights changed. Reviewing saved nonlinear MAP candidates"
        " for selective doubling gains within the factual budget.\n"
    )
```