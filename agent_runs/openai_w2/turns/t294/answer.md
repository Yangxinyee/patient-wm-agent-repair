The architecture confirms that this edit changes the recurrent state used by both factual predictions and dose rollouts. The recovery output was truncated before showing the cohort IDs, so I’ll retrieve those compactly and verify the saved fallback. Three disjoint remaining cohorts and an exact candidate match will confirm that validation can continue without changing weights.

```python
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

_r239_all_ids = set()
_r239_reference_groups = {}
for _r239_k in (34, 35, 36, 37):
    _r239_df = pd.read_csv(f"/work/fwd_{_r239_k}/factual_original.csv")
    _r239_staycol = next(c for c in _r239_df.columns if "stay" in c.lower())
    _r239_group = sorted(int(x) for x in _r239_df[_r239_staycol].unique())
    _r239_reference_groups[_r239_k] = _r239_group
    _r239_all_ids.update(_r239_group)

_r239_df44 = pd.read_csv("/work/fwd_44/factual_original.csv")
_r239_staycol44 = next(c for c in _r239_df44.columns if "stay" in c.lower())
_r239_done_ids = set(int(x) for x in _r239_df44[_r239_staycol44].unique())
_final_validation_ids = sorted(_r239_all_ids)
_r239_remaining = sorted(_r239_all_ids - _r239_done_ids)
_final_remaining_groups = [
    _r239_remaining[i:i+100] for i in range(0, len(_r239_remaining), 100)
]

assert len(_r239_all_ids) == 400
assert len(_r239_done_ids) == 100
assert _r239_done_ids <= _r239_all_ids
assert len(_r239_remaining) == 300

print("Forward44 columns:", list(_r239_df44.columns))
print("Forward44 first row:", _r239_df44.head(1).to_dict("records"))
print("Forward44 cohort:", len(_r239_done_ids),
      min(_r239_done_ids), max(_r239_done_ids))
for _r239_i, _r239_group in enumerate(_final_remaining_groups, 1):
    print(f"REMAINING_GROUP_{_r239_i} =", json.dumps(_r239_group))

_r239_current = np.asarray(load("core.b_in.weight"), dtype=np.float32)
print("Current SHA256 prefix:",
      hashlib.sha256(_r239_current.tobytes()).hexdigest()[:10])
print("Edited tensors:", edited())
for _r239_name in (
    "candidate_B237", "fallback_B237", "validated_B225",
    "delta_ins218", "delta_ne218"
):
    if _r239_name in globals():
        _r239_arr = np.asarray(globals()[_r239_name], dtype=np.float32)
        print(_r239_name, "shape", _r239_arr.shape,
              "hash", hashlib.sha256(_r239_arr.tobytes()).hexdigest()[:10])
        if _r239_arr.shape == _r239_current.shape:
            print("  max difference from current:",
                  float(np.max(np.abs(_r239_arr - _r239_current))))

with Path("/work/notes.md").open("a") as _r239_f:
    _r239_f.write(
        "\n\n## Run239: compact final-validation cohort recovery\n"
        "Recovered 400 reference tuning stays and the 100 already evaluated "
        "in forward44; stored three disjoint remaining groups of 100. "
        "Inspected current candidate and accepted-fallback fingerprints. "
        "No weights changed. Next complete paired factual validation "
        "before accepting the final gain-only trial.\n"
    )
```