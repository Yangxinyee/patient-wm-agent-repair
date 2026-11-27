The state audit shows that dose signals disappear after non-drug bundles, but norepinephrine recovery remains imperfect. I’ll check the signal where the value head reads it and test sensor-family separation. Held-out accuracy would support using this channel without imposing cross-drug effects.

```python
import numpy as _r140_np
import pandas as _r140_pd
import torch as _r140_torch
from sklearn.linear_model import Ridge as _r140_Ridge

if "_r140_features" not in globals():
    _r140_base_weights = {
        n: _r140_torch.as_tensor(load(n), dtype=_r140_torch.float32).clone()
        for n in NAMES
    }
    _r140_records = train[train.stay_id.isin(_r139_ids)].copy()
    _r140_oldW = W0
    try:
        W0 = _r140_base_weights
        _r140_cache = reading_cache(_r140_records, _r137_e)
        _r140_features = head_features_layout(
            _r140_cache, layout134, ageunit="seconds", nsamples=4, seed=913
        )
    finally:
        W0 = _r140_oldW

_r140_df = _r140_cache["df"]
_r140_wc_all = np.asarray(_r140_features["wc"])
_r140_wc = _r140_wc_all.mean(0)
_r140_pair = _r140_wc[:, [61, 15]] - _r140_wc[:, [21, 66]]
_r140_keys = pd.MultiIndex.from_arrays(
    [_r140_df.stay_id, np.floor(_r140_df.hour).astype(int) - 1],
    names=["stay_id", "t"]
)
_r140_doses = panel[["insulin", "norepinephrine"]].reindex(
    _r140_keys, fill_value=0
).fillna(0).to_numpy()
_r140_glucose = _r140_df.event.isin(["glucose_finger", "glucose_lab"]).to_numpy()
_r140_stays = _r140_df.stay_id.to_numpy()
_r140_unique = np.sort(np.unique(_r140_stays))
_r140_fitmask = np.isin(_r140_stays, _r140_unique[:70])
_r140_testmask = ~_r140_fitmask
_r140_scale = np.zeros(2)

print("=== HEAD-INPUT DOSE RECOVERY ===")
for _r140_j, _r140_target in [(0, _r140_glucose), (1, ~_r140_glucose)]:
    _r140_m = _r140_fitmask & _r140_target
    _r140_s = _r140_pair[:, _r140_j]
    _r140_scale[_r140_j] = (
        _r140_s[_r140_m] @ _r140_doses[_r140_m, _r140_j]
        / max(_r140_s[_r140_m] @ _r140_s[_r140_m], 1e-20)
    )
    _r140_prediction = _r140_s * _r140_scale[_r140_j]
    for _r140_label, _r140_mask in [
        ("fit", _r140_m),
        ("held-out", _r140_testmask & _r140_target)
    ]:
        print(
            ["insulin", "norepinephrine"][_r140_j], _r140_label,
            "scale", round(_r140_scale[_r140_j], 4),
            "RMSE", round(float(np.sqrt(np.mean(
                (_r140_prediction[_r140_mask]
                 - _r140_doses[_r140_mask, _r140_j])**2))), 5),
            "mean predicted/recorded",
            round(float(np.mean(_r140_prediction[_r140_mask])), 5),
            round(float(np.mean(_r140_doses[_r140_mask, _r140_j])), 5)
        )

_r140_X = _r140_wc.copy()
_r140_X[:, [61, 21, 15, 66]] = 0
_r140_gate = _r140_Ridge(alpha=1.0).fit(
    _r140_X[_r140_fitmask],
    np.where(_r140_glucose[_r140_fitmask], 1.0, -1.0)
)
_r140_gate_score = _r140_gate.predict(_r140_X)
_r140_signed = _r140_gate_score * np.where(_r140_glucose, 1.0, -1.0)
print("=== SENSOR-FAMILY SEPARATION ===")
for _r140_label, _r140_m in [
    ("fit", _r140_fitmask), ("held-out", _r140_testmask)
]:
    print(_r140_label,
          "accuracy", float(np.mean(_r140_signed[_r140_m] > 0)),
          "signed-margin quantiles",
          np.quantile(_r140_signed[_r140_m], [0, .01, .1, .5, 1]).round(4).tolist())
print("Reading count:", len(_r140_df))
print("Vocabulary centres:", VOCAB["value_bin_centres"])
```
