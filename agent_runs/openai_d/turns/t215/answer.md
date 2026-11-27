The stronger trial also passes factual fit on tuning stays 300–399: NLL rose by about 0.00275, below the 0.005 limit. That makes it a useful fallback, though it is not yet validated broadly. I’ll recover the dose-effect estimates and edit construction; proportional effects in the records would support fixing the weak half-dose response rather than lowering its target.

```python
from pathlib import Path
import numpy as np

head185 = [
    "attr_heads.value_bin.0.bias", "attr_heads.value_bin.0.weight",
    "attr_heads.value_bin.2.bias", "attr_heads.value_bin.2.weight"
]
c185_current = {n: load(n).clone() if hasattr(load(n), "clone") else load(n).copy()
                for n in head185}

text185 = Path("/work/notes.md").read_text()
start185 = text185.find("## Run 3\n")
print("RECORD-DERIVED EVIDENCE\n", text185[start185:start185+6400])
print("\nRECENT VARIABLE NAMES\n", [
    k for k in list(globals())
    if k.startswith(("c17", "c18", "r17", "r18"))
])
print("\nOTHER VARIABLE NAMES\n", [
    k for k in list(globals())
    if not k.startswith(("_", "r", "c")) and k not in ("np", "pd", "torch")
][-100:])
for k in ("c183_fallback", "c183_trial", "r171_combined_backup",
          "c182_trial", "c178_trial"):
    if k in globals():
        x = globals()[k]
        print("\n", k, ":", list(x) if isinstance(x, dict) else type(x).__name__)
with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 185: preserve stronger trial\nForward 24 factual on tuning 300..399: 1.64524 vs 1.64249, delta about +0.00275, below +0.005. Current stronger-NE four-tensor trial preserved in c185_current; no new weight change. Broader factual validation and half-dose asymmetry remain open.\n")
```