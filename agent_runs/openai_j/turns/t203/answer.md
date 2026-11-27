The sensor-based instrument is too weak to identify norepinephrine’s effect reliably, and the earlier calibration improved training loss but not held-out loss. Neither justifies a stronger edit. I’ll recover the remaining causal diagnostics and persistence estimates; agreement across credible record-based analyses would support the MAP target, while disagreement would warrant caution.

```python
from pathlib import Path
import pandas as pd
import numpy as np

text182 = Path("/work/notes.md").read_text()
print("CAUSAL NOTE CONTINUATION\n", text182[4500:11200])

for filename in ["dynamic_iv.csv", "proxy_iv.csv", "local_projections.csv"]:
    df182 = pd.read_csv("/work/" + filename)
    if "drug" in df182.columns:
        df182 = df182[df182["drug"].eq("norepinephrine")]
    print("\n", filename, "columns:", list(df182.columns))
    print(df182.head(7).to_string(index=False))

print("\nCURRENT LOCAL IMPLEMENTATION REFERENCES")
for name in [
    "trial158_gate", "wm_v151", "map164_fits",
    "wm_mc_train151", "wm_mc_tune151", "wm_train145", "wm_tune145"
]:
    obj = globals().get(name)
    if isinstance(obj, dict):
        print(name, "keys:", list(obj.keys())[:25])
    elif hasattr(obj, "shape"):
        print(name, tuple(obj.shape))
    elif obj is not None:
        print(name, repr(obj)[:180])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 182: evidence and calibration audit\n"
        "- Sensor-difference IV is weak for NE; not used as affirmative identification evidence.\n"
        "- Existing bin-bias calibration did not generalise to tuning; it is not a source of factual-loss budget.\n"
        "- Retained validated combined fallback and continued auditing record-based dynamic estimates.\n"
    )
```