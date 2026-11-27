The second factual check gives a +0.00270 NLL increase, consistent with the first check and still within the limit. I’ll now construct a MAP-only statistical correction using factual prior samples as constraints. It will be worth testing only if it increases norepinephrine sensitivity while keeping the combined loss increase below 0.005.

```python
import numpy as np, pandas as pd, torch, dis
from pathlib import Path

_pool = []
for _k in [15, 17]:
    _o = pd.read_csv(f"/work/fwd_{_k}/factual_original.csv")
    _e = pd.read_csv(f"/work/fwd_{_k}/factual_edited.csv")
    _pool.append(_e.nll.to_numpy() - _o.nll.to_numpy())
print("Pooled first-200 factual dNLL:", np.concatenate(_pool).mean())
print("Tilt vector:", wm_v151.cpu().numpy())
print("Immediate effects:", wm_beta151)
print("Train/CV split:", np.unique(wm_split151, return_counts=True))
print("Input scales:", wm_s152[:8], "range", wm_s152.min(), wm_s152.max())

_fit = wm_fits152[(1.0, 0.0001)]
_unscaled = np.r_[_fit[:192] / wm_s152, _fit[192]]
print("Insulin coefficient reconstruction RMSE:",
      np.sqrt(np.mean((trial158_coef - _unscaled)**2)))
print("Gate coefficient range/norm:",
      wm_gate153.min(), wm_gate153.max(), np.linalg.norm(wm_gate153))
print("Gate-input sample range:",
      (np.einsum("sni,i->sn", wm_train145["inputs"].numpy(), wm_gate153[:192])
       + wm_gate153[192]).min(),
      (np.einsum("sni,i->sn", wm_train145["inputs"].numpy(), wm_gate153[:192])
       + wm_gate153[192]).max())

print("\nLOCAL CONTRAST TARGETS:")
for _key, _cf in wm_cf146.items():
    _dr = np.asarray(_cf["delta_rate"])
    _active = np.asarray(_cf["active"], dtype=bool)
    _tau = wm_tau_targets151[_key]
    for _event in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
        _mask = _active & (wm_events151 == _event)
        if _mask.any():
            _shift = (_cf["mean"] - wm_train145["mean"]).numpy()[_mask]
            print(_key, _event, "n", _mask.sum(),
                  "rate change", round(float(_dr[_mask].mean()), 4),
                  "original shift", round(float(_shift.mean()), 4),
                  "target tilt", round(float(np.asarray(_tau)[_mask].mean()), 4))

print("\nFITTING FUNCTION DETAILS:")
for _name in ["wm_moments151", "wm_rawloss152"]:
    _fn = globals()[_name]
    print(_name, "names:", _fn.__code__.co_names)
    print("locals:", _fn.__code__.co_varnames)
    _ins = list(dis.get_instructions(_fn))
    print("bytecode:", " | ".join(
        f"{i.opname} {i.argrepr}" for i in _ins)[:2900])

with open("/work/notes.md", "a") as _f:
    _f.write("\n## Forward 17 and run 163: independent factual check\n"
             "- Second 100 tuning stays: original 1.61582, edited 1.61852, "
             "delta +0.00270; MAP unchanged.\n"
             "- Current insulin candidate passed factual checks on 200 stays.\n"
             "- Developing MAP-only NE correction with stochastic factual "
             "constraints; current edit remains the saved working fallback.\n")
```