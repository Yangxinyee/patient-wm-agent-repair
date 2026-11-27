The coefficients show that dose enters the first neuron in each pair; the earlier gate check inspected only the second, dose-free neuron. Thus it did not rule out clipping. The GPU summary also stopped on a column-name error. I’ll correct both checks before making a trial.

```python
from pathlib import Path
import numpy as np, torch, pandas as pd, inspect, dis, re

print("=== SIGNAL-BEARING NEURON AUDIT ===")
_X255 = torch.tensor(np.array(head_wc, copy=True), dtype=torch.float32)
_H255 = load("attr_heads.value_bin.0.weight")
_b255 = load("attr_heads.value_bin.0.bias")
_P255 = _X255 @ _H255.T + _b255
for _s, (_sig, _gate) in sensor_neurons.items():
    _ii255 = np.asarray(head_rows["event"] == _s)
    _raw255 = (_X255[:, _pd254] @ _H255[_sig, _pd254])[_ii255].numpy()
    print(_s,
          "signal activation", np.round(np.quantile(_P255[_ii255, _sig].numpy(), [0,.01,.1,.5,.9,.99,1]), 3),
          "raw private term", np.round(np.quantile(_raw255, [0,.01,.1,.5,.9,.99,1]), 3),
          "factual fraction below 0", float((_P255[_ii255, _sig] < 0).float().mean()))

print("\n=== GPU TRIAL SUMMARY ===")
for _dir in sorted(Path("/work").glob("fwd_*"), key=lambda p: int(p.name.split("_")[-1])):
    _parts = [_dir.name]
    _fp, _op = _dir/"factual_edited.csv", _dir/"factual_original.csv"
    if _fp.exists() and _op.exists():
        _fd, _od = pd.read_csv(_fp), pd.read_csv(_op)
        _parts.append(f"factual_delta={_fd['nll'].mean()-_od['nll'].mean():+.6f}")
    for _model in ["original", "edited"]:
        _dp = _dir / f"dose_edit_{_model}.csv"
        if _dp.exists():
            _dd = pd.read_csv(_dp)
            _rr = _dd[_dd["reached"].astype(str).str.lower().isin(["true", "1"])]
            _mh = _rr.groupby("edit")["shift"].mean()
            _ms = _rr.groupby(["edit", "stay_id"])["shift"].mean().groupby("edit").mean()
            _parts.append(_model + " hourly=" + str(_mh.round(3).to_dict()) +
                          " stay=" + str(_ms.round(3).to_dict()))
    if len(_parts) > 1:
        print(" | ".join(_parts))

print("\n=== DYNAMIC TARGET NOTES ===")
_sec255 = re.split(r"(?m)(?=^## )", Path("/work/notes.md").read_text())
_sel255 = [s for s in _sec255 if re.search(r"(persistence|autoregress|dynamic|rho|11\.893)", s, re.I)]
print("\n".join(s[:950] for s in _sel255[:4])[:3300])

print("\n=== HELPER CONSTANTS ===")
print("drug_atom205 constants:", drug_atom205.__code__.co_consts)
print("set205 constants:", set205.__code__.co_consts)
print("available definition strings:",
      [(k, len(v)) for k,v in dict(globals()).items()
       if isinstance(v,str) and len(v)>200 and re.search(r"(src|source|code|definition)", k, re.I)])
```