The earlier checks place most of the long-lag norepinephrine association in the first four ICU hours, outside the evaluated intervention period. The hour-12 signal remains unresolved. I’ll test it on disjoint training stays, including neighboring lags; replication would support a delayed effect, while an isolated failure to replicate would favor the null edit.

```python
import json
import time

print("events:", events)
print("nuisance settings:", settings_v3)
_v4_previous_groups = np.unique(stay_groups_v2[eligible])
_v4_start = int(_v4_previous_groups.max()) + 1
print("previous sample group range:",
      int(_v4_previous_groups.min()), int(_v4_previous_groups.max()))
print("replication group range:", _v4_start, n_stays - 1)
assert n_stays - _v4_start >= 8000, "Need a genuinely disjoint replication sample"

_v4_base = (
    (stay_groups_v2 >= _v4_start)
    & (A_ne_v2 > 0)
    & (treatment_hours_v2 >= 4)
    & np.isfinite(map_baseline_v3)
)
_v4_results = []
_v4_started = time.time()
_v4_map_channel = events.index("map_arterial")

for _lag in [1, 8, 10, 11, 12]:
    _target = future_reading_v2(_v4_map_channel, _lag) - map_baseline_v3
    _ar, _yr = matched_crossfit_v3(
        X_ne_v2, A_ne_v2, _target, _v4_base,
        f"disjoint NE lag {_lag}"
    )
    for _hmin in [4, 12]:
        _sel = _v4_base & (treatment_hours_v2 >= _hmin)
        _result = orthogonal_coefficient_v2(_ar, _yr, _sel)
        _result.update(
            horizon=_lag, hour_min=_hmin,
            adjustment="disjoint_late_matched_change",
            first_group=_v4_start
        )
        _v4_results.append(_result)
        print(json.dumps(_result))

print("elapsed seconds:", round(time.time() - _v4_started, 1))
with open("/work/notes.md", "a") as _f:
    _f.write("\n## Disjoint NE replication\n")
    _f.write(json.dumps(_v4_results) + "\n")
```