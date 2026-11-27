The dose encoding suggests a possible extrapolation problem, not evidence of a larger true effect. I’ll recover the encoder details and earlier causal checks before attempting a compensated edit. This approach is worth pursuing only if it can preserve observed-dose behavior while improving held-out rollouts.

```python
from pathlib import Path
import re
import numpy as np
import pandas as pd

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\nRECORD SCHEMA")
print(Path("/data/README.md").read_text())

print("\nRELEVANT TENSORS")
for name in NAMES:
    if any(k in name.lower() for k in
           ("num", "atom", "rate", "dose", "embed", "set_encoder")):
        print(name, SHAPES[name])

notes271 = Path("/work/notes.md").read_text()
sections271 = re.split(r"(?m)(?=^## )", notes271)

print("\nEARLIER IDENTIFICATION EVIDENCE")
evidence271 = [
    s for s in sections271
    if any(k in s.lower() for k in
           ("identification", "instrument", "randomization",
            "unconfounded", "dynamic iv", "dynamic-iv"))
]
for s in evidence271[:5]:
    print(s[:1300])

print("\nEARLIER ACTION-ENCODER EXPERIMENTS")
encoder271 = [
    s for s in sections271
    if any(k in s.lower() for k in
           ("numeric gain", "embedding", "atom encoder",
            "action-path", "action path", "numeric projection"))
]
for s in encoder271[-8:]:
    print(s[:1000])

print("\nSAVED CAUSAL ESTIMATES")
for filename in ("kernel_targets.csv", "sensor_iv.csv", "dynamic_iv.csv"):
    path = Path("/work") / filename
    if path.exists():
        table271 = pd.read_csv(path)
        print(filename, "shape", table271.shape)
        print(table271.head(8).to_string(index=False))

print("\nCACHED ACTION OBJECTS")
for key in ("wt", "repair_atoms", "repair_norms", "repair_codes",
            "cat_names", "post_av0", "post_e0", "atom_bins"):
    obj = globals().get(key)
    if isinstance(obj, dict):
        print(key, "dictionary keys:", list(obj)[:35])
    elif hasattr(obj, "shape"):
        print(key, "shape:", tuple(obj.shape))
    elif isinstance(obj, (list, tuple)):
        print(key, repr(obj)[:900])
    else:
        print(key, type(obj).__name__)

print("\nCURRENT EDITS", edited())
with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 271: compensated-encoder recovery\n"
        "- Read architecture and record schema; recovered earlier causal and "
        "action-encoder evidence.\n"
        "- Off-support categorical saturation is a hypothesis about model "
        "behavior, not evidence of the simulator's intervention effects.\n"
        "- No weights changed. Current calibrated candidate remains installed.\n"
        "- Counts: 271 Python runs, 34 forward passes.\n"
    )
```