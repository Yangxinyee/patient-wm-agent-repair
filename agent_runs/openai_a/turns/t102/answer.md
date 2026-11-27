The threshold analysis supports an insulin effect near −2 per unit, with no clear pre-treatment discontinuity. I’ll use that as a working target, subject to its identification assumptions. Next I’ll check event encoding and state timing; if the CPU reconstruction cannot match factual states closely, I will not use its synthetic contrasts.

```python
import numpy as np
import pandas as pd
from pathlib import Path

print("=== REMAINING COMPUTATION TENSORS ===")
for rr_n in NAMES:
    if not rr_n.startswith(("embed.", "encode.", "encode_action.", "core.", "attr_heads.")):
        print(rr_n, SHAPES[rr_n])

print("\n=== NORMALIZATION AND BIN TABLE STRUCTURE ===")
print("numeric type:", type(VOCAB["numeric"]).__name__)
print(repr(VOCAB["numeric"])[:3500])
print("atom kinds:", VOCAB.get("atom_kinds"))

print("\n=== EMBEDDING ROW NORMS ===")
for rr_n in NAMES:
    if rr_n.startswith("embed.") and rr_n.endswith(".weight") and SHAPES[rr_n][0] <= 17:
        rr_a = np.asarray(load_original(rr_n))
        print(rr_n, np.linalg.norm(rr_a, axis=1).round(3).tolist())

print("\n=== FIRST TRAIN STAY: RECORDS AND STATES ===")
rr_first_id = int(state_files[4]["stay_id"][0])
rr_raw_first = pd.read_csv("/data/train.csv", nrows=250)
rr_raw_first = rr_raw_first[rr_raw_first.stay_id == rr_first_id]
print("record columns:", rr_raw_first.columns.tolist())
print(rr_raw_first.head(24).to_string(index=False))
rr_sf = state_files[4]
rr_si = np.flatnonzero(rr_sf["stay_id"] == rr_first_id)
print(pd.DataFrame({k: np.asarray(rr_sf[k])[rr_si] for k in
                    ("hour", "has_insulin", "has_norepinephrine", "n_atoms")})
      .head(24).to_string(index=False))

print("\n=== ORIGINAL WEIGHT CACHE CHECK ===")
for rr_n in graft_names:
    if rr_n in W:
        rr_cached = W[rr_n].detach().cpu().numpy()
        print(rr_n, "max difference from original:",
              float(np.max(np.abs(rr_cached - np.asarray(load_original(rr_n))))))

print("\n=== CPU PART FUNCTION DETAILS ===")
for rr_n in ("cpu_head_parts", "repair_head_features", "reading_metadata"):
    rr_f = globals()[rr_n]
    print(rr_n, "globals:", rr_f.__code__.co_names)
    print("locals:", rr_f.__code__.co_varnames)
```