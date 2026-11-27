The recovered notes show history-adjusted drug effects estimated from the records, rather than assumed from medicine. The output was truncated before the saved-candidate details. I’ll retrieve those narrowly; a safer validated alternative would justify reconsidering the current edit, otherwise I’ll keep it and finish pooled rollout checks.

```python
from pathlib import Path
import numpy as np, pandas as pd, re

_notes = Path("/work/notes.md").read_text()
_sections = re.split(r"\n(?=## )", _notes)
print("=== RECORD IDENTIFICATION ===")
for _s in _sections:
    if re.match(r"## Run (6|7|8|9|10|11|12)\b", _s):
        print(_s[:720])

print("=== EXACT ARCHIVES ===")
for _fn in ["margin_validated400.npz", "margin_candidate.npz",
            "validated_current400.npz", "map_regularized_trial.npz"]:
    with np.load("/work/" + _fn, allow_pickle=False) as _a:
        print(_fn)
        for _k in _a.files:
            _v = _a[_k]
            print(" ", _k, _v.shape, str(_v.dtype),
                  repr(_v.tolist())[:200] if _v.size <= 8 else "")

print("=== TARGET AND EDIT INTERFACES ===")
_nprinted = 0
for _k in sorted(list(globals())):
    if not _k.startswith("_") and any(s in _k.lower() for s in
            ["target", "batch", "route", "margin", "map_add", "gluc_add"]):
        _v = globals()[_k]
        if isinstance(_v, np.ndarray):
            _desc = str((_v.shape, str(_v.dtype)))
        elif callable(_v):
            _desc = "callable"
        else:
            _desc = repr(_v)[:180]
        print(_k, _desc)
        _nprinted += 1
        if _nprinted >= 20:
            break

print("=== CURRENT ROLLOUT FILE SCHEMA ===")
for _folder in ["fwd_42", "fwd_43"]:
    _p = Path("/work") / _folder / "dose_edit_edited.csv"
    _df = pd.read_csv(_p)
    print(_folder, _df.shape, _df.columns.tolist())
    print(_df.head(2).to_string(index=False))

print("CURRENT EDITS:", edited())
with Path("/work/notes.md").open("a") as _f:
    _f.write("\n## Run 293: compact evidence recovery\n"
             "- Previous recovery output was truncated; retrieved selected identification notes, "
             "candidate metadata, and rollout schemas without changing weights.\n"
             "- Python runs: 293; forward passes: 42.\n")
```