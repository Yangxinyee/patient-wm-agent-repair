The replication attempt stopped before fitting: the earlier eligibility mask already spans all 30,000 training stays. There is no untouched training sample, so I won’t claim independent replication. I’ll instead check neighboring lags and stay-wise halves for stability; a consistent late effect would still challenge the null edit.

```python
_v5_base = (
    positive_ne_v2
    & (treatment_hours_v2 >= 4)
    & np.isfinite(map_baseline_v3)
)
print("eligible stays:", len(np.unique(stay_groups_v2[_v5_base])))
print("cross-fitting fold counts:", np.unique(row_folds_v2, return_counts=True))
print("positive NE rate quantiles:",
      np.quantile(A_ne_v2[_v5_base], [0, .25, .5, .75, .95, 1]).tolist())

_v5_results = []
_v5_residuals = {}
for _lag in [11, 12]:
    _target = future_reading_v2(events.index("map_arterial"), _lag) - map_baseline_v3
    _ar, _yr = matched_crossfit_v3(
        X_ne_v2, A_ne_v2, _target, _v5_base,
        f"NE stability lag {_lag}"
    )
    _v5_residuals[_lag] = (_ar, _yr)
    _late = _v5_base & (treatment_hours_v2 >= 12)
    for _label, _sel in [
        ("hours_ge4", _v5_base),
        ("hours_ge12", _late),
        ("hours_ge12_even_stays", _late & (stay_groups_v2 % 2 == 0)),
        ("hours_ge12_odd_stays", _late & (stay_groups_v2 % 2 == 1)),
        ("hours_ge12_below_max_rate", _late & (A_ne_v2 < np.max(A_ne_v2[_v5_base]))),
    ]:
        _result = orthogonal_coefficient_v2(_ar, _yr, _sel)
        _result.update(horizon=_lag, subset=_label,
                       adjustment="late_matched_change_stability")
        _v5_results.append(_result)
        print(json.dumps(_result))

with open("/work/notes.md", "a") as _f:
    _f.write(
        "\n## Replication correction and stability checks\n"
        "Run 28 stopped at an assertion before fitting or changing weights. "
        "Earlier eligibility spans all 30,000 training stays; no untouched "
        "training partition exists. Subsequent even/odd checks are stability "
        "diagnostics, not independent confirmatory evidence.\n"
    )
    _f.write(json.dumps(_v5_results) + "\n")
```