Sixteenfold gain saturated: insulin improved less than at fourfold gain, and norepinephrine remained near zero. I’ll return to gain four and examine dose-sensitive directions in the recurrent slots and latent state. A promising direction should respond to dose changes while having little variation in factual states; that would offer a more selective lever than further numeric amplification.

```python
import numpy as np, pandas as pd

clear_ops()
rank1("embed.numeric.weight", numeric_dose_axis, 3.0, side="in")

with open("/work/notes.md", "a") as f:
    f.write("\n## Forwards 11–12 / run 19: numeric saturation\n"
            "- Gain 16 factual increase +0.00246, but rollouts insulin -2.231/+1.531 and NE +0.157/+0.085.\n"
            "- Numeric gain saturates; restored gain 4 (-2.503/+1.691, NE +0.127/-0.003).\n"
            "- Investigating dose-sensitive recurrent/latent directions with low factual second moment.\n"
            "- Python runs: 19; forward passes: 12.\n")

print("=== STATE OBJECTS AND HEAD MATCHING ===")
for _name in ["sx", "states_original", "out_states", "state_for_reading"]:
    if _name in globals():
        print(_name, compact_structure(globals()[_name]))
for _event, _obj in head_surrogates.items():
    print(_event, "frame columns:", list(_obj["frame"].columns))
    print(_obj["frame"].head(2).to_string(index=False))

_state_x = np.load("/work/fwd_3/states_original.npz")["x"].astype(np.float64)
_parts_x = {
    "out": _state_x[:, :192],
    "hmean": _state_x[:, 192:960].reshape(-1, 4, 192).mean(axis=1),
    "z": _state_x[:, 960:]
}
_hslots = _state_x[:, 192:960].reshape(-1, 192)
repair_white = {}
repair_signal = {}

_hm = _parts_x["hmean"]
_oo = _parts_x["out"]
_hc = _hm - _hm.mean(axis=0)
_oc = _oo - _oo.mean(axis=0)
_h_to_out = np.linalg.solve(
    _hc.T @ _hc + 0.1*np.eye(192), _hc.T @ _oc)

for _j, _drug in enumerate(["insulin", "norepinephrine"]):
    repair_white[_drug] = {}
    repair_signal[_drug] = {}
    _coeff_parts = {
        "out": dose_state_coeff[_j, :192],
        "hmean": dose_state_coeff[_j, 192:960].reshape(4,192).mean(axis=0),
        "z": dose_state_coeff[_j, 960:]
    }
    for _part, _xx in _parts_x.items():
        _dd = np.asarray(_coeff_parts[_part], dtype=float)
        _cov_x = _hslots if _part == "hmean" else _xx
        _C = _cov_x.T @ _cov_x / len(_cov_x)
        _ridge = max(np.trace(_C)/len(_dd)*0.001, 1e-8)
        _vv = np.linalg.solve(_C + _ridge*np.eye(len(_dd)), _dd)
        _vv /= np.linalg.norm(_vv)
        repair_white[_drug][_part] = _vv
        _metrics = {
            "dose_coefficient_norm": float(np.linalg.norm(_dd)),
            "dose_projection": float(_vv @ _dd),
            "factual_projection_mean": float(np.mean(_cov_x @ _vv)),
            "factual_projection_sd": float(np.std(_cov_x @ _vv)),
            "dose_signal_over_factual_rms": float(
                abs(_vv @ _dd)/np.sqrt(np.mean((_cov_x @ _vv)**2)))
        }
        if _part in ["out", "hmean"]:
            for _event, _obj in head_surrogates.items():
                _ww = _obj["w"] if _part == "out" else _h_to_out @ _obj["w"]
                _metrics[_event+"_approx_response_per_alpha"] = float(
                    (_ww @ _vv)*(_vv @ _dd))
        repair_signal[_drug][_part] = _metrics
        print(_drug, _part, _metrics)

print("Direction cosines:",
      {p: float(repair_white["insulin"][p] @ repair_white["norepinephrine"][p])
       for p in ["out", "hmean", "z"]})
print("Current operations:", ops())
```
