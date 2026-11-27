“A” is arterial MAP, and the new moment candidates do not beat the active edit on that score, so I won’t install them. The earlier numerical-input scaling also failed in rollouts. I’ll instead screen stronger versions of the already validated MAP direction; useful headroom would mean a larger arterial response while factual loss stays safely below 0.005.

```python
# Algebraic scaling only; no gradients, optimiser, or weight installation.
_r224_direction = (map192_candidates[(10.0, 6.0)] - map192_cal) / 6.0
_r224_candidates = {}
_r224_rows = []

for _gain in [4.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0]:
    _coef = map192_cal + _gain * _r224_direction
    _weights = head177_make(_coef)
    _pred = head177_eval(wm_train145, _weights, wm_train145["centres"])
    _tune = head177_eval(wm_tune145, _weights, wm_tune145["centres"])
    _row = {
        "gain": _gain,
        "fit_dNLL": float((_pred["nll"] - orig_train209)[fitall208].mean()),
        "cv_dNLL": float((_pred["nll"] - orig_train209)[cvall208].mean()),
        "tune_dNLL": float((_tune["nll"] - orig_tune209).mean()),
        "coef_norm": float(np.linalg.norm(_coef)),
    }
    for _factor in [0.5, 2.0]:
        _mc = wm_cf146[("norepinephrine", _factor)]
        _cf = head177_eval(_mc, _weights, wm_train145["centres"])
        for _event, _short in [("map_arterial", "A"), ("map_cuff", "C")]:
            _mask = (cvall208 & np.asarray(_mc["active"], bool)
                     & (events206 == _event))
            _row[f"{_short}{_factor}"] = float(
                (_cf["mean"][_mask] - _pred["mean"][_mask]).mean()
                / np.asarray(_mc["delta_rate"])[_mask].mean())
    _r224_candidates[_gain] = {"coef": _coef.copy(), "weights": _weights}
    _r224_rows.append(_row)

_r224_table = pd.DataFrame(_r224_rows)
print(_r224_table.to_string(index=False))
print("\nActive edits remain:", edited())
_r224_table.to_csv("/work/map_gain_frontier224.csv", index=False)
with open("/work/notes.md", "a") as _f:
    _f.write("\n## Run 224: exact-head MAP gain frontier\n"
             "- Rejected moment218 candidates: arterial doubling sensitivity "
             "does not exceed the active edit.\n"
             "- Screened algebraic rescalings of the validated MAP direction "
             "using exact saved-head computations; no weights installed.\n"
             + _r224_table.to_string(index=False) + "\n")
```