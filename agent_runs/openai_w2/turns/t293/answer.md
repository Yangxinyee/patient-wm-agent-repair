The new edit increased factual NLL by about 0.00669 on this difficult cohort. That does not establish failure across all tuning stays, but the edit remains provisional. I’ll recover the cohort IDs and saved evidence before completing validation. Acceptance requires an overall increase below 0.005 and a better insulin rollout response.

```python
from pathlib import Path
import pandas as pd
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

print("\n=== EARLY NOTEBOOK: CAUSAL EVIDENCE ===")
_r238_notes = Path("/work/notes.md").read_text()
print(_r238_notes[:18000])

print("\n=== RELEVANT PERSISTENT VARIABLES ===")
_r238_snapshot = list(globals().items())
_r238_terms = (
    "candidate", "fallback", "validated", "delta_ins", "delta_ne",
    "tuning", "cohort", "ids", "target", "slope", "persist", "rho"
)
for _r238_name, _r238_value in sorted(_r238_snapshot):
    if _r238_name.startswith("_r238"):
        continue
    if any(t in _r238_name.lower() for t in _r238_terms):
        _r238_shape = getattr(_r238_value, "shape", None)
        if _r238_shape is not None:
            print(_r238_name, type(_r238_value).__name__, tuple(_r238_shape))
        elif isinstance(_r238_value, (str, int, float, bool, type(None))):
            print(_r238_name, repr(_r238_value)[:300])
        elif isinstance(_r238_value, (list, tuple, dict, set)):
            print(_r238_name, type(_r238_value).__name__,
                  "length", len(_r238_value), repr(_r238_value)[:400])

print("\n=== FACTUAL FILE SCHEMAS AND COHORTS ===")
for _r238_k in (34, 35, 36, 37, 44):
    _r238_path = Path(f"/work/fwd_{_r238_k}/factual_original.csv")
    _r238_df = pd.read_csv(_r238_path)
    print("forward", _r238_k, "shape", _r238_df.shape,
          "columns", list(_r238_df.columns))
    print(_r238_df.head(2).to_string(index=False))
    for _r238_col in _r238_df.columns:
        if "stay" in _r238_col.lower():
            _r238_ids = sorted(_r238_df[_r238_col].dropna().unique().tolist())
            print("stay column", _r238_col, "count", len(_r238_ids),
                  "first", _r238_ids[:3], "last", _r238_ids[-3:])

print("\n=== CURRENT EDITS ===")
print(edited())

with Path("/work/notes.md").open("a") as _r238_f:
    _r238_f.write(
        "\n\n## Run238: final gain-only trial recovery\n"
        "Forward44 difficult-cohort factual increase is approximately "
        "+0.00669; this cohort alone does not establish the full-tuning result. "
        "Recovered architecture, record schema, earlier causal evidence, "
        "persistent candidate variables and factual cohort IDs. "
        "No weights changed. Candidate_B237 remains provisional; "
        "fallback_B237 remains the accepted fallback.\n"
    )
```