This candidate fails the factual constraint: NLL increased by 0.005355, slightly above the allowed 0.005. Glucose-lab predictions account for much of the loss. I’ll inspect earlier candidates and calibration attempts before choosing a safer edit; any replacement must preserve useful dose responses without exceeding the factual limit.

```python
import inspect
from pathlib import Path
import numpy as np
import pandas as pd
import re

print("=== FACTUAL MEAN BIASES ===")
for _m271, _d271 in _f270.items():
    _s271 = _d271.assign(error=_d271.predicted_mean-_d271.observed)
    print(_m271)
    print(_s271.groupby("event").agg(
        n=("nll", "size"), nll=("nll", "mean"),
        mean_bias=("error", "mean"), prediction=("predicted_mean", "mean")
    ).round(6).to_string())

print("\n=== REBALANCING FUNCTION ===")
_fn271 = globals().get("install_rebalanced268")
if callable(_fn271):
    try:
        print(inspect.getsource(_fn271))
    except Exception:
        print("names:", _fn271.__code__.co_names)
        print("constants:", str(_fn271.__code__.co_consts)[:1600])

print("\n=== INSULIN AND NE SAVED CONTRACTS ===")
for _file271 in ["insulin_candidate63.npz", "insulin_trial219.npz",
                 "ne_candidate151.npz"]:
    with np.load("/work/"+_file271, allow_pickle=False) as _a271:
        print(_file271)
        for _key271 in _a271.files:
            _v271 = _a271[_key271]
            print(_key271, _v271.shape,
                  _v271.tolist() if _v271.size <= 8 else
                  {"norm": round(float(np.linalg.norm(_v271)), 5)})

print("\n=== COMPACT RETAINED TARGETS ===")
for _key271 in ["targets69", "record_schedule_target69", "bootstrap_targets76",
                "targets214", "blend219", "insulin_scale236",
                "insulin_blend236", "ne_scale236"]:
    if _key271 in globals():
        _v271 = globals()[_key271]
        if isinstance(_v271, pd.DataFrame):
            print(_key271, _v271.shape, _v271.head(8).to_string(index=False))
        elif isinstance(_v271, np.ndarray):
            print(_key271, _v271.shape,
                  _v271.tolist() if _v271.size <= 20 else "array omitted")
        else:
            print(_key271, str(_v271)[:1000])

print("\n=== PRIOR CALIBRATION / SAFETY NOTES ===")
_text271 = Path("/work/notes.md").read_text()
_lines271 = [x for x in _text271.splitlines()
             if re.search(r"compensat|calibrat|Forward2[0-9]|Forward3[0-5]|cost=|lab-safe",
                          x, re.I)]
print("\n".join(_lines271[-22:])[:3500])

print("\n=== AVAILABLE HEAD / STATE HELPERS ===")
print([k for k,v in list(globals().items())
       if not k.startswith("_") and callable(v) and
       any(t in k.lower() for t in ["head", "read", "value", "state", "install", "factual"])])

with Path("/work/notes.md").open("a") as _log271:
    _log271.write(
        "\n## Run 271: joint268 rejected on factual fit\n"
        "Forward34 original NLL=1.620435206, edited=1.625789773, "
        "delta=.005354567 (fails .005). By-event degradation: "
        "glucose_finger=.005925, glucose_lab=.018507, map_arterial=.003620, "
        "map_cuff=.001913. Bootstrap interval [.002196,.008817]. "
        "Recovering safer candidate and prior calibration contracts. "
        "joint268 remains installed only for diagnostics, not accepted. "
        "Python runs: 271; forwards: 35.\n")
```