The retained objects include both threshold-based and dose-innovation estimates, so we can inspect the causal evidence without rebuilding it. I’ll compare their estimates and the lag-response tables. Agreement would support a target effect; unstable or contradictory estimates would argue against choosing an edit yet.

```python
import numpy as np
import pandas as pd
from pathlib import Path

pd.set_option("display.max_columns", 14)
pd.set_option("display.width", 160)
print("=== THRESHOLD ESTIMATES ===")
print(rd_refined.to_string(index=False))
print("=== IMMEDIATE INNOVATION ESTIMATES ===")
print(dml_results.to_string(index=False))
print("=== LONGITUDINAL MOMENTS ===")
for drug, tab_imp55 in impulse_tables_repair.items():
    print(drug)
    print(tab_imp55.to_string(index=False))
print("=== PROVISIONAL RESPONSE KERNELS ===")
for drug, kernel55 in provisional_kernels.items():
    print(drug, kernel55)
print("=== PERSISTENCE FITS ===")
for drug, fit55 in persistence_repair.items():
    print(drug, fit55)
print("=== BASELINE ROLLOUT ===")
print(base_roll.groupby("edit").agg(
    mean_shift=("shift", "mean"), n=("shift", "size")).to_string())
print("=== HOURLY PANEL ===")
print(pan.head(7).to_string())
print("=== EARLY NOTEBOOK ===")
print(Path("/work/notes.md").read_text()[:3200])
with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 55\nRetrieved retained threshold, innovation and longitudinal-moment diagnostics to select an evidence-based intervention target. No weights edited.\n")
```