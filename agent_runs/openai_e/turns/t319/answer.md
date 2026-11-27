The factual means do not show a simple bias that a constant correction could fix, and earlier calibration attempts found little headroom. I’ll check the final record-derived targets and validation cohorts, then reduce the recurrent edit’s strength. The smaller edit must pass factual validation while retaining most of its dose sensitivity.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import re

_text272 = Path("/work/notes.md").read_text()
print("=== FINAL EARLY CAUSAL IDENTIFICATION ===")
_sections272 = re.split(r"(?=^## )", _text272, flags=re.M)
for _s272 in _sections272:
    if re.match(r"## (?:Run|Runs) (?:69|70|71|72|73|74|75|76|77|78)\b", _s272):
        print(_s272[:900])

print("\n=== LIVE CAUSAL PARAMETERS ===")
for _k272, _v272 in list(globals().items()):
    if _k272.startswith("_"):
        continue
    if re.search(r"beta|rho|persist", _k272, re.I):
        if np.isscalar(_v272) and not isinstance(_v272, str):
            print(_k272, _v272)
        elif isinstance(_v272, np.ndarray) and _v272.size <= 12:
            print(_k272, _v272.tolist())
print("bootstrap_targets76 quantiles:",
      np.quantile(bootstrap_targets76, [.025, .5, .975]).tolist()
      if "bootstrap_targets76" in globals() else "unavailable")

for _name272 in ["record_schedule_target69", "targets214"]:
    _fn272 = globals().get(_name272)
    if callable(_fn272):
        print(_name272, "arguments",
              _fn272.__code__.co_varnames[:_fn272.__code__.co_argcount],
              "defaults", _fn272.__defaults__,
              "names", _fn272.__code__.co_names,
              "constants", str(_fn272.__code__.co_consts)[:600])

print("\n=== RECENT FACTUAL COHORTS ===")
for _j272 in [24, 26, 27, 28, 30, 31, 33, 34]:
    _p272 = Path(f"/work/fwd_{_j272}")
    _oa272, _ea272 = _p272/"factual_original.csv", _p272/"factual_edited.csv"
    if _oa272.exists() and _ea272.exists():
        _od272, _ed272 = pd.read_csv(_oa272), pd.read_csv(_ea272)
        print(_j272, "IDs", int(_od272.stay_id.min()), int(_od272.stay_id.max()),
              "stays", _od272.stay_id.nunique(),
              "delta", round(float(_ed272.nll.mean()-_od272.nll.mean()), 8))
print("\n=== BASELINE CONTRACT ===")
with np.load("/work/baseline77.npz", allow_pickle=False) as _a272:
    for _k272 in _a272.files:
        _v272 = _a272[_k272]
        print(_k272, _v272.shape,
              _v272.tolist() if _v272.size <= 5 else
              "norm="+str(round(float(np.linalg.norm(_v272)), 4)))

print("\n=== DOSE CSV SCHEMA ===")
_d272 = pd.read_csv("/work/fwd_35/dose_edit_edited.csv")
print(list(_d272.columns))
print(_d272.head(2).to_string(index=False))

with Path("/work/notes.md").open("a") as _log272:
    _log272.write(
        "\n## Run 272: recover final causal targets and cohort alignment\n"
        "Current joint268 factual mean drift is not a simple calibration error; "
        "prior aligned temperature calibration gave no meaningful headroom. "
        "Reviewing final identification and measured validation cohorts before "
        "conservative recurrent-strength reduction. No weights changed. "
        "Python runs: 272; forwards: 35.\n")
```