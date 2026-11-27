This message repeats the factual audit and contains no new rollout result. I’ll check whether the rollout files were produced before spending another forward pass. Complete files would let me assess the candidate; missing files would mean the request still needs to run. No weights will change.

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd

_r241_dir = Path("/work/fwd_48")
_r241_expected = [
    _r241_dir / "dose_edit_original.csv",
    _r241_dir / "dose_edit_edited.csv"
]
print("Forward48 directory exists:", _r241_dir.exists())
print("Forward48 files:",
      sorted(p.name for p in _r241_dir.iterdir()) if _r241_dir.exists() else [])
print("Edited tensors:", edited())
print("Current matches candidate_B237:",
      bool(np.array_equal(np.asarray(load("core.b_in.weight")),
                          np.asarray(candidate_B237))))

_final_rollout_forward = None
_r241_results = []
if all(p.exists() for p in _r241_expected):
    _final_rollout_forward = 48
    _final_gain_rollout = pd.read_csv(_r241_expected[1])
    _r241_original = pd.read_csv(_r241_expected[0])
    for _r241_label, _r241_df in (
        ("original", _r241_original),
        ("current", _final_gain_rollout),
        ("fallback", pd.read_csv("/work/fwd_38/dose_edit_edited.csv"))
    ):
        _r241_mask = _r241_df["reached"].astype(str).str.lower().isin(["true", "1"])
        for _r241_edit, _r241_g in _r241_df.loc[_r241_mask].groupby("edit"):
            _r241_s = _r241_g.groupby("stay_id")["shift"].agg(["mean", "sum", "size"])
            _r241_mean = float(_r241_g["shift"].mean())
            _r241_res = _r241_s["sum"] - _r241_mean * _r241_s["size"]
            _r241_n = len(_r241_s)
            _r241_se = float(
                np.sqrt(_r241_n / (_r241_n - 1) * np.sum(_r241_res ** 2))
                / len(_r241_g)
            ) if _r241_n > 1 else float("nan")
            _r241_results.append({
                "model": _r241_label, "edit": _r241_edit,
                "reached_hours": len(_r241_g), "stays": _r241_n,
                "hour_weighted_shift": _r241_mean,
                "hour_weighted_cluster_se": _r241_se,
                "equal_stay_shift": float(_r241_s["mean"].mean())
            })
    print("ROLLOUT COMPARISON:")
    print(json.dumps(_r241_results, indent=2))
    _r241_status = "Forward48 files present; recovered rollout comparison."
else:
    _r241_status = "Forward48 dose files absent/incomplete; no new rollout evidence."
    print(_r241_status)

with Path("/work/notes.md").open("a") as _r241_f:
    _r241_f.write(
        "\n\n## Run241: duplicate-result recovery\n"
        + _r241_status + "\n"
        "No weights changed; candidate_B237 remains provisional.\n"
        + (json.dumps(_r241_results) + "\n" if _r241_results else "")
    )
```