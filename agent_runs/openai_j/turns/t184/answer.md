The shared MAP fit gives only modest safe improvement. The required logit shifts differ substantially between arterial and cuff readings, so a shared correction may be an unnecessary restriction. I’ll test separate sensor-gated regressions; stronger arterial sensitivity at the same factual-loss cost would support that approach.

```python
import numpy as np, pandas as pd, torch
from scipy.linalg import solve

print("Closest direct insulin-fit matches:")
_matches = []
for _key, _w in wm_fits152.items():
    _matches.append((_key, float(np.sqrt(np.mean(
        (trial158_coef - trial158_gain * np.asarray(_w))**2)))))
print(sorted(_matches, key=lambda x: x[1])[:3])

sensor165_fits, sensor165_rows = {}, []
sensor165_tune_events = wm_tune_meta145["event"].to_numpy()
for _sensor in ["map_arterial", "map_cuff"]:
    _family = wm_events151 == _sensor
    _train = _family & map164_train_mask
    _F = map164_F[:, _train].reshape(-1, 193)
    _K = _F.T @ _F / len(_F)
    _C = np.zeros((193, 193))
    _b = np.zeros(193)
    for _key, _cf in wm_cf146.items():
        _mask = np.asarray(_cf["active"], dtype=bool) & _train
        _D = ((_cf["inputs"].numpy().astype(np.float64) - map164_X)
              [:, _mask] / map164_scale).reshape(-1, 192)
        _D = np.column_stack([_D, np.zeros(len(_D))])
        _weight = 0.5 if _key[0] == "norepinephrine" else 0.05
        _C += _weight * (_D.T @ _D / len(_D))
        if _key[0] == "norepinephrine":
            _y = np.tile(np.asarray(wm_tau_targets151[_key])[_mask],
                         map164_X.shape[0])
            _b += _weight * (_D.T @ _y / len(_D))
    for _fw in [0.03, 0.1, 0.3, 1.0, 3.0]:
        _R = np.eye(193) * 1e-4
        _R[-1, -1] = 1e-8
        _scaled = solve(_C + _fw * _K + _R, _b, assume_a="pos")
        _raw = np.r_[_scaled[:192] / map164_scale, _scaled[192]]
        for _gain in [1.0, 2.0, 4.0]:
            _w = _gain * _raw
            sensor165_fits[(_sensor, _fw, _gain)] = _w
            _row = {"sensor": _sensor, "fw": _fw, "gain": _gain}
            for _label, _mc, _mask in [
                ("train", wm_train145, _family),
                ("tuning", wm_tune145, sensor165_tune_events == _sensor)
            ]:
                _nll, _mu, _tau, _p = wm_rawloss152(_mc, _w, _mask)
                _delta = (_nll - _mc["nll"]).mean(0).numpy()
                if _label == "train":
                    _row["cv_dNLL"] = float(_delta[~map164_train_mask].mean())
                else:
                    _row["tuning_dNLL"] = float(_delta.mean())
                    _row["tilt_rms"] = float(_tau.square().mean().sqrt())
                    _row["latent_sd"] = float(
                        _tau[:, _mask].std(0).square().mean().sqrt())
            _base = map164_mean(wm_train145, _w, _family,
                                wm_train145["centres"])
            for _factor in [0.5, 2.0]:
                _cf = wm_cf146[("norepinephrine", _factor)]
                _mu = map164_mean(_cf, _w, _family,
                                  wm_train145["centres"])
                _mask = np.asarray(_cf["active"], dtype=bool) & _family
                _dr = np.asarray(_cf["delta_rate"])[_mask]
                _row[f"N{_factor}"] = float(
                    np.dot(_dr, (_mu - _base)[_mask]) / np.dot(_dr, _dr))
            sensor165_rows.append(_row)

sensor165_frontier = pd.DataFrame(sensor165_rows)
sensor165_frontier.to_csv("/work/map_sensor_frontier.csv", index=False)
print(sensor165_frontier.to_string(index=False,
                                  float_format=lambda x: f"{x:.5f}"))

sensor165_labels = ["glucose_finger", "glucose_lab",
                    "map_arterial", "map_cuff"]
_gateF = map164_F.reshape(-1, 193)
_gateY = np.tile(np.column_stack([
    np.where(wm_events151 == s, 1.0, -1.0) for s in sensor165_labels
]), (map164_X.shape[0], 1))
_gateR = np.eye(193) * 1e-6
_gateR[-1, -1] = 1e-8
_gateScaled = solve(_gateF.T @ _gateF / len(_gateF) + _gateR,
                    _gateF.T @ _gateY / len(_gateF), assume_a="pos")
sensor165_gates = {
    s: np.r_[_gateScaled[:192, j] / map164_scale, _gateScaled[192, j]]
    for j, s in enumerate(sensor165_labels)
}
print("\nONE-SENSOR GATE MARGINS:")
for _sensor, _w in sensor165_gates.items():
    for _label, _mc, _events in [
        ("train", wm_train145, wm_events151),
        ("tuning", wm_tune145, sensor165_tune_events)
    ]:
        _g = _mc["inputs"].numpy() @ _w[:192] + _w[192]
        print(_sensor, _label, "positive min", _g[:, _events == _sensor].min(),
              "negative max", _g[:, _events != _sensor].max())

with open("/work/notes.md", "a") as _f:
    _f.write("\n## Run 165: sensor-specific MAP feasibility\n"
             "- Shared MAP correction showed limited safe local response; "
             "testing separate arterial/cuff tilts and linear sensor gates.\n"
             "- No new edits installed. Factual and rollout fallback retained.\n")
```