The notebook shows that the original rollouts were nearly insensitive to dose, and that the current repair uses isolated dose channels plus paired value-head neurons. I’ll check whether those neurons clip large negative perturbations and summarize the GPU trials consistently. Clipping would support changing the gates; otherwise, the next trial should address transport or calibration.

```python
from pathlib import Path
import numpy as np, torch, pandas as pd, inspect, dis, re

print("=== PAIRED HEAD COEFFICIENTS ===")
_H254 = load("attr_heads.value_bin.0.weight")
_b254 = load("attr_heads.value_bin.0.bias")
_O254 = load("attr_heads.value_bin.2.weight")
_pd254 = [d for p in private_pairs for d in p]
for _s, (_ref, _ctl) in sensor_neurons.items():
    print(_s, "ref/control", (_ref, _ctl),
          "bias", np.round(_b254[[_ref, _ctl]].numpy(), 4),
          "private coefficients", np.round(_H254[[_ref, _ctl]][:, _pd254].numpy(), 3),
          "other-row norms", np.round(
              torch.linalg.vector_norm(_H254[[_ref, _ctl]][:, [i for i in range(192) if i not in _pd254]], dim=1).numpy(), 3),
          "output norms", np.round(torch.linalg.vector_norm(_O254[:, [_ref, _ctl]], dim=0).numpy(), 4))
if "head_wc" in globals():
    _pre254 = torch.as_tensor(head_wc, dtype=torch.float32) @ _H254.T + _b254
    for _s, (_ref, _ctl) in sensor_neurons.items():
        _ix254 = np.asarray(head_rows.event == _s)
        print(_s, "reconstructed control quantiles",
              np.round(np.quantile(_pre254[_ix254, _ctl].numpy(), [0, .01, .1, .5, .9, .99, 1]), 3))

print("\n=== GPU TRIAL SUMMARY ===")
for _dir in sorted(Path("/work").glob("fwd_*"), key=lambda p: int(p.name.split("_")[-1])):
    _parts = [_dir.name]
    _fp = _dir / "factual_edited.csv"
    _op = _dir / "factual_original.csv"
    if _fp.exists() and _op.exists():
        _fd, _od = pd.read_csv(_fp), pd.read_csv(_op)
        _parts.append("factual_delta=" + str(round(float(_fd.nll.mean()-_od.nll.mean()), 6)))
    for _model in ["original", "edited"]:
        _dp = _dir / f"dose_edit_{_model}.csv"
        if _dp.exists():
            _dd = pd.read_csv(_dp)
            _rr = _dd[_dd.reached.astype(str).str.lower().isin(["true", "1"])]
            _mh = _rr.groupby("edit").shift.mean()
            _ms = _rr.groupby(["edit", "stay_id"]).shift.mean().groupby("edit").mean()
            _parts.append(_model + " hourly=" + str(_mh.round(3).to_dict()) +
                          " stay=" + str(_ms.round(3).to_dict()))
    if len(_parts) > 1:
        print(" | ".join(_parts))

print("\n=== DYNAMIC TARGET NOTES ===")
_sec254 = re.split(r"(?m)(?=^## )", Path("/work/notes.md").read_text())
_sel254 = [s for s in _sec254 if re.search(r"(persistence|autoregress|dynamic|rho|11\.893)", s, re.I)]
print("\n".join(s[:1100] for s in _sel254[:4])[:3000])

print("\n=== DRUG ATOM HELPER ===")
try:
    print(inspect.getsource(drug_atom205)[:2400])
except (OSError, TypeError):
    print(dis.Bytecode(drug_atom205).dis()[:2400])
```