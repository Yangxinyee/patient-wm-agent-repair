The second cohort also gives exactly zero for all four edited shifts. That confirms the intended invariance, while the record-based analyses favor negligible magnitude effects over the unstable nonzero associations. I’ll keep the edit and audit the final files and pooled factual NLL. Mismatched rows, unintended weight changes, or an NLL increase above 0.005 would prevent submission.

```python
import json
import numpy as np
import pandas as pd
from pathlib import Path

_o = factual_frames["original"].reset_index(drop=True)
_e = factual_frames["edited"].reset_index(drop=True)
_keys = ["stay_id", "hour", "event", "observed", "forward_file"]
assert _o[_keys].equals(_e[_keys])
assert _o["stay_id"].nunique() == 400
assert len(_o) == len(_e)

_audit = _o[["stay_id", "hour", "event", "nll"]].copy()
_audit["edited_nll"] = _e["nll"].to_numpy()
_audit["delta"] = _audit["edited_nll"] - _audit["nll"]
_delta = float(_audit["delta"].mean())
_cluster_sums = _audit.groupby("stay_id")["delta"].sum()
_cluster_counts = _audit.groupby("stay_id").size()
_scores = _cluster_sums - _delta * _cluster_counts
_G = len(_scores)
_cluster_se = float(np.sqrt(_G / (_G - 1) * np.sum(_scores.to_numpy() ** 2)) / len(_audit))
final_factual_audit = {
    "n_stays": 400,
    "n_readings": len(_audit),
    "original_nll": float(_audit["nll"].mean()),
    "edited_nll": float(_audit["edited_nll"].mean()),
    "delta_nll": _delta,
    "stay_cluster_se": _cluster_se,
    "approximate_95pct_upper_delta": _delta + 1.96 * _cluster_se,
    "allowed_delta": 0.005
}
print("=== POOLED FACTUAL AUDIT ===")
print(json.dumps(final_factual_audit, indent=2))
print("=== BY EVENT ===")
print(_audit.groupby("event")[["nll", "edited_nll", "delta"]].mean().to_string())

_expected = {"embed.bins.dose.weight", "embed.bins.rate.weight", "embed.numeric.weight"}
assert set(edited()) == _expected
for _name in sorted(_expected):
    _a = np.asarray(load(_name))
    _b = np.asarray(load_original(_name))
    assert _a.shape == _b.shape and np.isfinite(_a).all()
    if "bins." in _name:
        assert np.array_equal(_a[0], _b[0])
        assert np.array_equal(_a[1:], np.repeat(_a[1:2], 16, axis=0))
    else:
        assert np.array_equal(_a[:, :2], _b[:, :2])
        assert np.all(_a[:, [2, 4]] == 0)

print("=== ROLLOUT FILE AUDIT ===")
for _k in (5, 10):
    _df = pd.read_csv(f"/work/fwd_{_k}/dose_edit_edited.csv")
    print("forward directory", _k, "columns", _df.columns.tolist())
    for _c in _df.columns:
        if "shift" in _c.lower() and pd.api.types.is_numeric_dtype(_df[_c]):
            _finite = _df[_c].dropna().to_numpy()
            if len(_finite):
                print(_c, "max_absolute", float(np.max(np.abs(_finite))))
                assert np.all(_finite == 0)

with open("/work/notes.md", "a") as _f:
    _f.write("\n## Final candidate audit / run 19 and forward 8\n")
    _f.write("Kept candidate A unchanged, hash 7971f330d6. Independent second100 "
             "rollout cohort also gave exact zero for insulin_x2, insulin_half, "
             "ne_x2 and ne_half. Original second100 shifts were "
             "+0.141, -0.218 mg/dL and -0.002, +0.008 mmHg respectively.\n")
    _f.write("Pooled factual audit: " + json.dumps(final_factual_audit) + "\n")
    _f.write("Training-only evidence: rich-history cross-fits, late-period analyses, "
             "multiple horizons, independent sensors, and insulin policy-threshold "
             "checks favor negligible dose-magnitude effects. Isolated nonlinear "
             "subset signals were not stable across broader samples/specifications; "
             "they were not extrapolated to doubled doses. This supports a null "
             "magnitude repair, not a claim that observational records prove "
             "absence of every possible drug effect. Drug identity/presence and "
             "measurement-value channels remain intact. No model gradients or "
             "weight optimizer used. Expected submitted readouts: all four zero.\n")
print("Audit complete; edits unchanged:", sorted(edited()))
```
