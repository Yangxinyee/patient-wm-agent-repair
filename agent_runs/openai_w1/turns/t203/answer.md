Increasing the posterior correction still barely changed the rollouts, so that pathway is not delivering the needed dose response. I’ll inspect the direct action-to-state path and the latest factual result. A useful replacement must transmit dose changes while keeping factual NLL within the allowed bound.

```python
from pathlib import Path
import pandas as pd
import numpy as np

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\nRELEVANT TENSORS")
for name in NAMES:
    if any(s in name.lower() for s in
           ["prelude", "action", "slot", "transition", "recur", "gru", "readout", "value", "head", "z_prior"]):
        print(name, SHAPES[name])

print("\nFORWARD 15")
for p in sorted(Path("/work/fwd_15").glob("*")):
    print(p.name)
    if p.suffix == ".csv":
        df15 = pd.read_csv(p)
        print("shape", df15.shape, "columns", list(df15.columns))
        print(df15.head(2).to_string(index=False))
        ncols = [c for c in df15.columns if "nll" in c.lower()]
        if ncols:
            print(df15[ncols].mean(numeric_only=True).to_dict())

print("\nLIVE HELPERS AND TARGETED CACHES")
for key, value in list(globals().items()):
    if key.startswith(("rp_", "r168_", "rr_", "ds_", "causal", "kernel", "lag")):
        if callable(value):
            print(key, "callable")
        elif isinstance(value, dict):
            print(key, "dict keys", list(value)[:18])
        elif hasattr(value, "shape"):
            print(key, type(value).__name__, tuple(value.shape))
        elif isinstance(value, (str, int, float, list, tuple)):
            print(key, str(value)[:220])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 181\nGain-200 posterior probe still yields negligible rollout changes (forward 16). "
            "This rejects gain amplification of the present posterior pathway as a useful repair. "
            "Recovering direct action/state architecture and factual forward 15 before replacing it. "
            "Counts: 181 Python runs, 16 forwards.\n")
```