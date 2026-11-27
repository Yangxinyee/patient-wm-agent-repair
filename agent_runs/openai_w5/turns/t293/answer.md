The private dose channels reset within seconds, so multi-hour treatment effects must pass through the model’s ordinary reading feedback. That is a plausible limitation, not yet a demonstrated cause. I’ll compare the hourly rollout responses with the record-derived response curve and identify the saved states needed for a reversible persistence trial.

```python
from pathlib import Path
import numpy as np, pandas as pd, torch

_t257 = pd.read_csv("/data/tuning.csv")
_d257 = pd.read_csv("/work/fwd_18/dose_edit_edited.csv")
_ids257 = sorted(_d257["stay_id"].unique())
_dr257 = _t257[_t257["stay_id"].isin(_ids257) &
               _t257["event"].isin(["insulin","norepinephrine"])].copy()
_dr257["start"] = np.floor(_dr257["hour"] + 1e-6).astype(int)
_rates257 = _dr257.groupby(["stay_id","start","event"])["rate"].sum().to_dict()
_params257 = {"insulin": (-1.8381467583, .8762725203),
              "norepinephrine": (3.5998780750, .6335817172)}
_target257 = {}
for _sid in _ids257:
    for _drug, (_beta,_phi) in _params257.items():
        _eff = 0.0
        for _hh in range(13,25):
            _eff = _phi*_eff + _beta*_rates257.get((_sid,_hh-1,_drug),0.0)
            _target257[(_sid,_hh,_drug)] = _eff
_d257["drug"] = np.where(_d257["edit"].str.startswith("insulin"),"insulin","norepinephrine")
_d257["multiplier_delta"] = np.where(_d257["edit"].str.endswith("x2"),1.0,-.5)
_d257["estimated_effect"] = [
    _target257.get((int(r.stay_id),int(r.hour),r.drug),np.nan)*r.multiplier_delta
    for r in _d257.itertuples()
]
_r257 = _d257[_d257["reached"].astype(str).str.lower().isin(["true","1"])]
print("=== HOURLY MODEL / RECORD-DERIVED ESTIMATE ===")
print(_r257.groupby(["edit","hour"])[["shift","estimated_effect"]].mean().round(3).to_string())
print("\n=== STAY-WEIGHTED ESTIMATES ===")
print(_r257.groupby(["edit","stay_id"])[["shift","estimated_effect"]].mean()
      .groupby("edit").mean().round(4).to_string())
print("\n=== STATE / READING ALIGNMENT ===")
print("head_rows IDs", sorted(head_rows["stay_id"].unique())[:4],
      "n", head_rows["stay_id"].nunique())
print("state_ids", np.asarray(state_ids)[:5], "n", len(state_ids))
for _p in ["/work/fwd_13/states_edited.npz","/work/fwd_15/states_edited.npz"]:
    _sf = np.load(_p)
    print(_p, {k: _sf[k].shape for k in _sf.files})
    for _k in _sf.files:
        if "stay" in _k:
            print(_k, np.unique(_sf[_k])[:5])
        if _k == "x":
            _hh = _sf[_k][:,192:960].reshape(-1,4,192)
            print("ordinary slot SD median", float(np.median(_hh.std(axis=2))),
                  "private magnitude", np.quantile(np.abs(_hh[:,:, _pd256]), [.5,.9,.99]).round(7))
print("\n=== AVAILABLE STATE OBJECTS ===")
for _k,_v in sorted(dict(globals()).items()):
    if isinstance(_v,dict) and {"x","hour"}.issubset(_v):
        print(_k, {kk: np.shape(vv) for kk,vv in _v.items()})
with Path("/work/notes.md").open("a") as _f:
    _f.write("\n## Runs 252–257: persistence audit\n"
             "- Stronger glucose readout alone gives little rollout gain and costs factual NLL; retained baseline strengths.\n"
             "- Factual reconstructed signal neurons are near +8, not GELU-clipped.\n"
             "- Private core.flow.a=0 resets dose coordinates within seconds. Multi-hour persistence therefore relies on ordinary autoregressive feedback.\n"
             "- Comparing hourly GPU effects with record-derived AR response before a reversible persistent-channel trial; estimates remain assumption-dependent.\n")
```