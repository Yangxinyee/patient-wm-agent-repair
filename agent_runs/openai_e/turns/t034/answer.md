We now have the exact posterior tensors and saved states. I’ll check which reconstructed drug bundles actually match those states, then test dose-innovation recovery only on verified rows. Strong held-out recovery would justify a posterior edit; poor recovery would mean we need a different representation.

```python
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

print("=== ENCODING CONSTANTS ===")
print("event kinds:", VOCAB["event_kinds"])
print("atom kinds:", VOCAB["atom_kinds"])
print("units:", RECON_UNITS, "rate units:", RECON_RATE_UNITS)
print("ekinds:", RECON_EKINDS)
print("numeric:", {str(k): (v["median"], v["iqr"]) for k, v in NUM_SPEC.items()})
print("dynamic parameters:", dyn_fit["x"])
print("policy models:", {k: type(v).__name__ for k, v in dose_policy.items()})

_P31 = np.asarray(recon_post_mean)
_H31 = np.asarray(recon_post_hidden)
_Z31 = S0["x"][:, 960:]
_err31 = np.sqrt(np.mean((_P31 - _Z31)**2, axis=1))
_groups31 = {}
for _i31 in range(len(_err31)):
    _key31 = (int(S0["stay_id"][_i31]), int(round(S0["hour"][_i31] * 60)))
    _starts31 = record_starts.get(_key31, [])
    _ends31 = record_ends.get(_key31, [])
    _label31 = (
        "+".join(sorted(str(r.event) for r in _starts31)) or "none",
        "+".join(sorted(str(r.event) for r in _ends31)) or "none"
    )
    _groups31.setdefault(_label31, []).append(_i31)

print("=== POSTERIOR MATCH BY BUNDLE (largest groups) ===")
for _label31, _ix31 in sorted(_groups31.items(), key=lambda x: -len(x[1]))[:16]:
    _ee31 = _err31[_ix31]
    print(_label31, "n", len(_ix31),
          "median/max", np.round([np.median(_ee31), np.max(_ee31)], 6),
          "verified", int((_ee31 < 0.002).sum()))

print("=== EXISTING ACTION TARGETS ===")
print("folds:", np.unique(action_fold, return_counts=True))
print("raw SD:", np.std(action_raw, axis=0),
      "innovation SD:", np.std(action_targets, axis=0))
print("first rows:", [
    (int(j), round(float(S0["hour"][j]), 4),
     np.round(action_raw[i], 3).tolist(),
     np.round(action_targets[i], 3).tolist(),
     round(float(_err31[j]), 6))
    for i, j in enumerate(action_rows[:8])
])

_good31 = _err31[action_rows] < 0.002
_sid31 = S0["stay_id"][action_rows]
_train31 = _good31 & (_sid31 % 5 != 0)
_test31 = _good31 & (_sid31 % 5 == 0)
print("verified train/test:", int(_train31.sum()), int(_test31.sum()))
_results31 = []
if _train31.sum() > 200 and _test31.sum() > 50:
    _hh31 = _H31[action_rows].astype(np.float64)
    _mu31 = _hh31[_train31].mean(axis=0)
    _sd31 = np.maximum(_hh31[_train31].std(axis=0), 0.05)
    _xx31 = (_hh31 - _mu31) / _sd31
    for _alpha31 in (0.1, 1.0, 10.0, 100.0):
        _rr31 = Ridge(alpha=_alpha31).fit(
            _xx31[_train31], action_targets[_train31])
        _pr31 = _rr31.predict(_xx31[_test31])
        _row31 = {
            "alpha": _alpha31,
            "innovation_R2": np.round(r2_score(
                action_targets[_test31], _pr31,
                multioutput="raw_values"), 4).tolist(),
            "innovation_RMSE": np.round(np.sqrt(np.mean(
                (_pr31-action_targets[_test31])**2, axis=0)), 4).tolist()
        }
        _results31.append(_row31)
    print("=== VERIFIED POSTERIOR INNOVATION PROBES ===")
    print(_results31)

with open("/work/notes.md", "a") as _f31:
    _f31.write(
        "\n## Run 31: verified posterior feature test\n"
        "Audited posterior reconstruction by bundle type and tested cross-stay "
        "innovation recovery on rows with posterior RMSE below .002.\n"
        f"Verified train/test: {_train31.sum()}/{_test31.sum()}; "
        f"probe results: {_results31}.\n"
        "No weights edited. Python runs: 31; forward passes: 3.\n"
    )
```