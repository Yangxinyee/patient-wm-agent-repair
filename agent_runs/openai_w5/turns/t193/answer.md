The centering regressions use ordinary features that may themselves encode the current dose. That could suppress within-range dose changes while leaving larger changes less canceled—a possible explanation for the weak halving responses. I’ll inspect the local state-step implementation so I can test this directly with a fixed history. The hypothesis is supported if the baseline-dose estimate follows half-dose edits much more closely than doubled doses.

```python
import numpy as np, torch

print("=== LOCAL STATE-STEP IMPLEMENTATION ===")
_ret = compact161(wm_step_saved, maxchars=6100)
if _ret is not None:
    print(str(_ret)[:6100])

print("\n=== SAVED-CONTEXT IMPLEMENTATION ===")
_ret = compact161(context_from_saved, maxchars=1700)
if _ret is not None:
    print(str(_ret)[:1700])

print("\n=== FACTUAL PRIVATE DOSE AND CENTERING ===")
for _sp in _r148_specs:
    _event = _sp["event"]
    _mask = np.asarray(_r148_event_array) == _event
    _d = _sp["drug"]
    _cc = np.array(_r137_coords[2*_d:2*_d+2])
    _wc = np.asarray(_r147_wc).mean(axis=0)
    _raw = _wc[:, _cc] @ (_sp["row_signal"][_cc] / _sp["c"])
    _proxy = _sp["proxy"].predict(_wc)
    _actual = np.asarray(_r147_dose)[:, _d]
    print(_event,
          "n", int(_mask.sum()),
          "actual/raw/proxy means", np.round([_actual[_mask].mean(), _raw[_mask].mean(), _proxy[_mask].mean()], 4),
          "raw_rmse", round(float(np.sqrt(np.mean((_raw[_mask]-_actual[_mask])**2))), 4),
          "proxy_rmse", round(float(np.sqrt(np.mean((_proxy[_mask]-_actual[_mask])**2))), 4))

print("\n=== ROLLOUT TABLE SCHEMA ===")
print(dose151.head(3).to_string(index=False))
print("hour-13 rows")
print(dose151.loc[np.isclose(dose151["hour"], 13)].groupby("edit").mean(numeric_only=True).to_string()
      if "hour" in dose151 and "edit" in dose151 else list(dose151.columns))
```
