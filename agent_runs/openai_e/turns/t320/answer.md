The corrected record analysis supports positive norepinephrine effects of roughly +4–4.6 mmHg for doubling, so the current +0.593 response remains too small. I’ll compare the saved norepinephrine candidates and calculate targets for this exact cohort. That will show whether another tested direction is preferable to simply shrinking the current edit.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import re

print("=== EXACT-COHORT RECORD-DERIVED TARGETS ===")
for _j273 in [29, 35]:
    _p273 = f"/work/fwd_{_j273}/dose_edit_edited.csv"
    if Path(_p273).exists():
        print("forward", _j273, targets214(_p273))

print("\n=== SAVED NE CANDIDATE SIZES ===")
for _fn273 in ["ne_candidate82.npz", "ne_candidate148.npz",
               "ne_candidate151.npz", "ne_candidate154.npz",
               "ne_candidate155.npz", "ne_candidate199.npz"]:
    with np.load("/work/"+_fn273, allow_pickle=False) as _a273:
        print(_fn273, [
            (k, _a273[k].shape,
             _a273[k].tolist() if _a273[k].size <= 3 else
             round(float(np.linalg.norm(_a273[k])), 4))
            for k in _a273.files])

print("\n=== MEASURED NE FRONTIER ===")
_rows273 = []
for _j273 in range(11, 24):
    _dir273 = Path(f"/work/fwd_{_j273}")
    _row273 = {"forward": _j273}
    _op273, _ep273 = _dir273/"factual_original.csv", _dir273/"factual_edited.csv"
    if _op273.exists() and _ep273.exists():
        _oo273, _ee273 = pd.read_csv(_op273), pd.read_csv(_ep273)
        _row273["NLL_delta"] = _ee273.nll.mean()-_oo273.nll.mean()
    _dp273 = _dir273/"dose_edit_edited.csv"
    if _dp273.exists():
        _dd273 = pd.read_csv(_dp273)
        _rr273 = _dd273.reached.astype(str).str.lower().isin(["true", "1"])
        _row273.update(_dd273.loc[_rr273].groupby("edit")["shift"].mean().to_dict())
    if len(_row273) > 1:
        _rows273.append(_row273)
print(pd.DataFrame(_rows273).round(6).to_string(index=False))

print("\n=== NE DIRECTION SELECTION NOTES ===")
_text273 = Path("/work/notes.md").read_text()
for _section273 in re.split(r"(?=^## )", _text273, flags=re.M):
    if re.match(r"## (?:Run|Runs) (?:148|149|150|151|152|153|154|155|196|197|198|199|200|201)\b",
                _section273):
        print(_section273[:650])

print("\n=== CURRENT COMPOSITION BASES ===")
for _key273 in ["base268", "balanced268", "ne268", "original_b268"]:
    if _key273 in globals():
        _v273 = np.asarray(globals()[_key273])
        print(_key273, _v273.shape, round(float(np.linalg.norm(_v273)), 5))

with Path("/work/notes.md").open("a") as _log273:
    _log273.write(
        "\n## Run 273: cohort targets and measured NE frontier\n"
        "Recovered corrected record-only NE dynamics: beta≈3.17, phi≈.704; "
        "early NE2 reached-hour target≈4.2–4.6, half≈-2.1–-2.3. "
        "Inspecting exact current-cohort targets and previously measured candidates "
        "before final safety-oriented trials. No weights changed. "
        "Python runs: 273; forwards: 35.\n")
```