The stronger state penalty also fails: NLL increased by 0.01126, essentially unchanged. The local penalty is not predicting full-sequence behavior reliably. I’ll test a 40% version of the correction whose rollout we measured, and recover earlier validation results. It is useful only if it passes factual fit and improves on the previous feasible edit.

```python
import numpy as np
import pandas as pd
from pathlib import Path
import re

candidate155 = (
    original154 + baseline_delta154 + 0.4 * ne151_rebuilt154
).astype(np.float32)
save_edit("core.b_in.weight", candidate155)
np.savez("/work/ne_candidate155.npz",
         weight=candidate155, baseline_delta=baseline_delta154,
         ne_delta=0.4 * ne151_rebuilt154, strength=0.4)
print("Installed baseline77 + 0.4 × candidate151 NE correction.")

print("\n=== HISTORICAL FACTUAL NLL ===")
for folder155 in sorted(Path("/work").glob("fwd_*"),
                        key=lambda p: int(p.name.split("_")[-1])):
    for model155 in ["original", "edited"]:
        file155 = folder155 / f"factual_{model155}.csv"
        if file155.exists():
            df155 = pd.read_csv(file155)
            cols155 = [c for c in df155.columns if "nll" in c.lower()]
            if cols155:
                print(folder155.name, model155,
                      {c: round(float(df155[c].mean()), 6) for c in cols155})

print("\n=== INSULIN BASELINE DEVELOPMENT ===")
notes155 = Path("/work/notes.md").read_text()
for section155 in re.split(r"(?=^## )", notes155, flags=re.M):
    header155 = section155.splitlines()[0] if section155 else ""
    if re.search(r"\bRun (58|59|63|64|66|68|70|72|75|76|77)\b", header155):
        print(section155[:550])

print("\n=== AVAILABLE DECODER HELPERS ===")
for nm155 in sorted(list(globals())):
    obj155 = globals()[nm155]
    if callable(obj155) and re.search(r"(read|head|logit|decode|view)", nm155, re.I):
        print(nm155)

with open("/work/notes.md", "a") as f155:
    f155.write(
        "\n## Forward 19 / Run 155: state penalty fails; direct shrinkage trial\n"
        "Gamma=1 candidate factual NLL=1.64587 (+.01126), rejected. "
        "Installed original + verified baseline77 + .4*NE_candidate151_delta, "
        "saved as ne_candidate155.npz. Recovery of historical factual results and insulin baseline evidence "
        "accompanies this reversible, unaccepted trial. Python runs: 155; forwards: 19.\n"
    )
```