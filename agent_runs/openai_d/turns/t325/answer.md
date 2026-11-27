The recovered notes show that adjusting for prior readings reverses the apparent insulin association, giving a stable next-hour effect near −2 per rate unit. Norepinephrine estimates are more sensitive to the controls, so I should not treat the preliminary estimate as settled. The output was truncated; I’ll extract the later causal conclusions and rollout targets. Stable estimates across identification checks would support the repair’s intended effects.

```python
import numpy as np
import pandas as pd
import re
from pathlib import Path

_n = Path("/work/notes.md").read_text()
_start = _n.find("## Run 4")
print("=== EARLY CAUSAL ROBUSTNESS CHECKS ===")
print(_n[_start:_start + 5200])

print("=== LATER TARGET / CAUSAL CONCLUSION LINES ===")
_lines = _n.splitlines()
_hits = [
    i for i, s in enumerate(_lines)
    if re.search(
        r"causal target|rollout target|target shifts|targets:|"
        r"accepted causal|true effect|structural coefficient|"
        r"record-derived target|estimated truth|identified effect",
        s, re.I
    )
]
_budget = 2200
for _i in _hits:
    _text = "\n".join(_lines[max(0, _i-1):_i+3])
    if len(_text) <= _budget:
        print(_text)
        _budget -= len(_text)

print("=== RELEVANT VARIABLES ===")
print(sorted(k for k in globals()
             if not k.startswith("_")
             and re.search(r"caus|target|effect|coef|candidate|stay|truth", k, re.I)))

print("=== ROLLOUT FILE SCHEMA ===")
_r43 = pd.read_csv("/work/fwd_43/dose_edit_edited.csv")
print(_r43.shape, _r43.columns.tolist())
print(_r43.head(2).to_string(index=False))

print("=== NEXT COHORT IDS ===")
_tuning_ids_recovered = sorted(
    pd.read_csv("/data/tuning.csv", usecols=["stay_id"])["stay_id"].unique().tolist()
)
print(_tuning_ids_recovered[100:200])
```
