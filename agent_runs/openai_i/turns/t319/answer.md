The earlier lag analyses estimated roughly −11 for doubled insulin and +3.8 for doubled norepinephrine on one subset; these are estimates, not supplied truth. The fallback had matched factual NLL, and the 80%-strength candidate is now provisional. I’ll check the installed weights and recover the exact validation stays before testing whether the smaller response meets the factual limit.

```python
import numpy as np
from pathlib import Path

print("VALIDATION_STAYS", dev_ids278)
print("EDITED", edited())
print("GAIN278_INSTALLED", all(np.array_equal(load(k),v)
      for k,v in cand278.items()))
print("CAUSAL_TARGETS215", repr(globals().get("causal_targets215"))[:1400])

n279 = Path("/work/notes.md").read_text()
for marker in ["## Python 29;", "## Python 30;", "## Python 150", "## Python 215", "## Python 269"]:
    p = n279.find(marker)
    if p >= 0:
        print("\n", n279[p:p+1100])
print("\nZ_IN_RESPONSE_COLUMNS", [
    (j, round(float(np.linalg.norm(c278["z_in.weight"][:,j]
                                   -b278["z_in.weight"][:,j])),5))
    for j in range(32)
    if np.linalg.norm(c278["z_in.weight"][:,j]
                      -b278["z_in.weight"][:,j]) > 1e-6])
print("CANDIDATE_DIFF_NORMS", {
    k:round(float(np.linalg.norm(cand278[k] -
        (b278[k] if k in b278 else load_original(k)))),4)
    for k in cand278})
with open("/work/notes.md","a") as f:
    f.write("\n## Python 279 — compact validation recovery\n"
            "- Checked provisional gain278 and recovered exact factual-validation stays.\n"
            "- Record-based cumulative targets require cohort-specific comparison, not medical assumptions.\n")
```
