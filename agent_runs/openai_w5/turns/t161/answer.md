Factual fit leaves room for a small dose-innovation correction, but I have not yet verified that the paired differences represent dose accurately. I’ll compare them with recorded rates and inspect their decay. Poor dose recovery or incorrect timing would refute this channel design.

```python
import pandas as _r139_pd
import numpy as _r139_np

print("=== AVAILABLE DATAFRAMES ===")
print([(k, v.shape) for k, v in list(globals().items())
       if not k.startswith("_") and isinstance(v, _r139_pd.DataFrame)])
print("=== PANEL SCHEMA ===")
print(panel.index.names, panel.columns.tolist())
print(panel.head(3).to_string())
print("=== CHANNEL WEIGHTS ===")
_r139_R = [61, 21, 15, 66]
for _r139_n in ["core.flow.a", "core.delta_in", "core.out_norm.weight",
                "core.out_norm.bias", "query_norm.weight", "query_norm.bias"]:
    print(_r139_n, load(_r139_n)[_r139_R].tolist())
print("State sample:", list(zip(
    _r137_e["stay_id"][:12].tolist(), _r137_e["hour"][:12].tolist(),
    _r137_e["has_insulin"][:12].tolist(), _r137_e["has_norepinephrine"][:12].tolist()
)))
print("W0 type and sample:", type(W0).__name__,
      type(W0["core.flow.a"]).__name__)

_r139_ids = set(_r137_e["stay_id"].tolist())
_r139_parts = []
for _r139_chunk in _r139_pd.read_csv(
    "/data/train.csv", usecols=["stay_id", "hour", "event", "rate"], chunksize=250000
):
    _r139_parts.append(_r139_chunk[
        _r139_chunk.stay_id.isin(_r139_ids)
        & _r139_chunk.event.isin(["insulin", "norepinephrine"])
    ])
_r139_drugs = _r139_pd.concat(_r139_parts, ignore_index=True)
_r139_rate_table = _r139_drugs.pivot_table(
    index=["stay_id", "hour"], columns="event", values="rate", aggfunc="sum"
).fillna(0)
_r139_keys = _r139_pd.MultiIndex.from_arrays(
    [_r137_e["stay_id"], _r137_e["hour"]], names=["stay_id", "hour"]
)
_r139_rates = _r139_rate_table.reindex(
    _r139_keys, fill_value=0
).reindex(columns=["insulin", "norepinephrine"], fill_value=0).to_numpy()
_r139_drugmask = (_r137_e["has_insulin"] | _r137_e["has_norepinephrine"]).astype(bool)
_r139_outdiff = _r137_e["x"][:, [61, 15]] - _r137_e["x"][:, [21, 66]]
_r139_hdiff = (_r137_e["x"][:, 192:960].reshape(-1, 4, 192).mean(1)[:, [61, 15]]
              - _r137_e["x"][:, 192:960].reshape(-1, 4, 192).mean(1)[:, [21, 66]])
print("=== DOSE RECOVERY ON DRUG BUNDLES ===")
for _r139_label, _r139_diff in [("out", _r139_outdiff), ("mean h", _r139_hdiff)]:
    _r139_X = _r139_np.c_[np.ones(_r139_drugmask.sum()),
                           _r139_diff[_r139_drugmask]]
    _r139_Y = _r139_rates[_r139_drugmask]
    _r139_coef = _r139_np.linalg.lstsq(_r139_X, _r139_Y, rcond=None)[0]
    _r139_pred = _r139_X @ _r139_coef
    print(_r139_label, "coefficients", _r139_coef.round(5).tolist(),
          "RMSE", np.sqrt(np.mean((_r139_pred - _r139_Y)**2, axis=0)).round(5).tolist(),
          "rate ranges", [_r139_Y.min(0).tolist(), _r139_Y.max(0).tolist()])
print("Non-drug pair-difference max:",
      np.max(np.abs(_r139_outdiff[~_r139_drugmask]), axis=0).tolist())
print("=== SMALL FUNCTION METADATA ===")
for _r139_name in ["rw125", "channel_targets105", "mechanistic_shifts"]:
    _r139_fn = globals().get(_r139_name)
    if _r139_fn:
        print(_r139_name, _r138_inspect.signature(_r139_fn),
              "globals", _r139_fn.__code__.co_names,
              "constants", repr(_r139_fn.__code__.co_consts)[:500])
```
