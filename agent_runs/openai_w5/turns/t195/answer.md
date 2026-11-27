The CPU reconstruction does not match the saved GPU states closely enough, so I won’t treat its dose responses as evidence yet. I’ll check whether the error affects the private dose coordinates or only the ordinary recurrent state, and inspect the attention conventions. Accurate private coordinates would still permit a narrower transport test.

```python
import numpy as np, torch

print("=== NON-HEAD WEIGHT CONSISTENCY ===")
_diffs170 = []
for _n in NAMES:
    if _n.startswith("attr_heads.value_bin"):
        continue
    _a = torch.as_tensor(load(_n))
    _b = torch.as_tensor(_r147_clean_weights[_n])
    _d = float(torch.max(torch.abs(_a-_b)))
    if _d > 1e-7:
        _diffs170.append((_n, _d))
print(_diffs170[:20], "count", len(_diffs170))

print("\n=== PRIVATE VERSUS ORDINARY RECONSTRUCTION ===")
_oldW, _oldS = W0, s4
_rows170 = []
try:
    W0, s4 = _r169_current, _r169_states
    _cc = np.array(_r137_coords)
    _ordinary = np.setdiff1d(np.arange(192), _cc)
    for _rec in _r169_tests[:4]:
        _j = _rec["j"]
        _truth = torch.as_tensor(s4["x"][_j, 192:960]).reshape(1, 4, 192)
        for _inc in [False, True]:
            _pred = wm_step_saved(_j, include_current=_inc)
            _rows170.append({
                "j": _j, "include_current": _inc,
                "truth_rms": float(torch.sqrt(torch.mean(_truth**2))),
                "ordinary_rmse": float(torch.sqrt(torch.mean((_pred[..., _ordinary]-_truth[..., _ordinary])**2))),
                "private_rmse": float(torch.sqrt(torch.mean((_pred[..., _cc]-_truth[..., _cc])**2))),
                "true_I_pair": float((_truth[..., _cc[0]]-_truth[..., _cc[1]]).mean()),
                "pred_I_pair": float((_pred[..., _cc[0]]-_pred[..., _cc[1]]).mean())
            })
finally:
    W0, s4 = _oldW, _oldS
print(pd.DataFrame(_rows170).to_string(index=False))

print("\n=== LAYER IMPLEMENTATION ===")
_ret = compact161(wm_layer, maxchars=1900)
if _ret is not None:
    print(str(_ret)[:1900])
print("\n=== ATTENTION IMPLEMENTATION ===")
_ret = compact161(wm_attention, maxchars=4600)
if _ret is not None:
    print(str(_ret)[:4600])
```
