The rollout refutes the local calibration: factual fit held, but insulin sensitivity weakened and doubled norepinephrine moved MAP slightly downward. I’ll stop using the reconstructed value heads as evidence. Instead, I’ll build a small statistical readout from official predictions and states; it must explain held-out official predictions before guiding another weight edit.

```python
from pathlib import Path
import pandas as pd
import numpy as np
import inspect

print("=== OFFICIAL FACTUAL TABLE ===")
official194 = pd.read_csv("/work/fwd_11/factual_original.csv")
print("shape:", official194.shape, "columns:", official194.columns.tolist())
print(official194.head(5).to_string(index=False))

print("\n=== AVAILABLE OFFICIAL STATES ===")
state_files194 = sorted(Path("/work").glob("fwd_*/states_*.npz"))
for p in state_files194:
    with np.load(p) as a:
        print(str(p), {k:a[k].shape for k in a.files})
        for k in a.files:
            if a[k].ndim == 1:
                print(" ", k, a[k][:5].tolist())

print("\n=== EXACT CORE INTERFACE ===")
for k in ["r59_core", "repaired_core_local"]:
    fn = globals().get(k)
    if fn:
        print(k, "args", list(inspect.signature(fn).parameters),
              "uses", fn.__code__.co_names,
              "locals", fn.__code__.co_varnames[:25])
for k in ["context151", "state_keys_audit", "bundle_records_audit",
          "future_index59", "observed59", "rr35_pre", "r59_context"]:
    v = globals().get(k)
    if isinstance(v,dict):
        print(k, {a: (type(b).__name__, getattr(b,"shape",None))
                  for a,b in list(v.items())[:12]})
    elif v is not None:
        print(k, type(v).__name__, getattr(v,"shape",None), "length",
              len(v) if hasattr(v,"__len__") else None)

print("\n=== ONE TUNING STAY'S EVENT ORDER ===")
tune194 = pd.read_csv("/data/tuning.csv")
print(tune194[tune194.stay_id == tuning_stays193[0]][
    ["hour","event","value","rate"]].head(28).to_string(index=False))

with Path("/work/notes.md").open("a") as f:
    f.write("\n\n## Forwards 11–12 / Run 194: reject reconstructed-head calibration\n"
            "- Candidate193 official factual delta -0.00045 nats/reading.\n"
            "- Official rollout: insulin x2 -1.183, half +0.520; NE x2 -0.049, half -0.365. Worse than baseline despite promising local-head slopes.\n"
            "- Therefore local reconstructed value heads are not a reliable dose-response surrogate.\n"
            "- Starting a statistical readout calibrated directly to official original predictions and states; no new weights changed.\n"
            "- Python runs: 194; forward passes: 12.\n")
```