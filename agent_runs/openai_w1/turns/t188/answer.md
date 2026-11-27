The records support insulin lowering glucose and norepinephrine raising MAP, with similar signs across measurement channels. A dose-minus-policy correction may preserve factual fit because factual dose deviations are small. I’ll check whether the posterior’s final layer can express that correction; good held-out decoding and stronger responses to dose edits would support it.

```python
import numpy as np, torch

print("=== CURRENT POSTERIOR RECONSTRUCTION ===")
_ztruth = np.asarray(ds_s["x"])[:, -32:]
for _label, _inp in (("ds_in", ds_in), ("rr_input4", rr_input4)):
    _t = torch.as_tensor(_inp, dtype=torch.float32)
    for _actname, _act in (("gelu", torch.nn.functional.gelu), ("silu", torch.nn.functional.silu)):
        _hidden = _act(_t @ ds_w0.T + ds_b0)
        _pred = (_hidden @ ds_w2.T + ds_b2).numpy()
        _scores = []
        for _sel in (slice(0,32), slice(32,64), slice(0,64,2), slice(1,64,2)):
            _scores.append(round(float(np.sqrt(np.mean((_pred[:,_sel]-_ztruth)**2))),6))
        print(_label, _actname, "RMSE first/last/even/odd:", _scores)
print("=== DECODERS ===")
for _key, _obj in ds_decoders.items():
    print(_key, type(_obj).__name__)
    if isinstance(_obj, dict):
        print({str(_k): (tuple(_v.shape) if hasattr(_v,"shape") else type(_v).__name__) for _k,_v in _obj.items()})
    elif hasattr(_obj, "coef_"):
        print("coef", np.shape(_obj.coef_), "intercept", _obj.intercept_)
print("=== PHYSICAL DIRECTIONS ===")
print("container:", type(r168_physdirections).__name__)
for _key, _obj in r168_physdirections.items():
    print(_key, type(_obj).__name__)
    if isinstance(_obj, dict):
        for _k,_v in _obj.items():
            if isinstance(_v,(np.ndarray,torch.Tensor)):
                print(" ",_k,tuple(_v.shape),"norm",round(float(np.linalg.norm(np.asarray(_v))),6))
            elif isinstance(_v,(float,int,str)):
                print(" ",_k,_v)
            else:
                print(" ",_k,type(_v).__name__)
    elif isinstance(_obj,(np.ndarray,torch.Tensor)):
        print("shape",tuple(_obj.shape),"norm",float(np.linalg.norm(np.asarray(_obj))))
print("=== RECENT CACHE NAMES ===")
for _k,_v in sorted(list(globals().items())):
    if _k.startswith(("r167","r168","fresh","ctx167")):
        print(_k, type(_v).__name__, tuple(_v.shape) if hasattr(_v,"shape") else (list(_v.keys())[:5] if isinstance(_v,dict) else ""))
print("=== BUNDLE COVERAGE ===")
print("rows",len(ds_rates),"active counts",(ds_rates>0).sum(axis=0),
      "held-out rows",int(np.asarray(ds_hold,dtype=bool).sum()),
      "rate percentiles",np.percentile(ds_rates,[0,50,90,99,100],axis=0))
with open("/work/notes.md","a") as _f:
    _f.write("\n## Run 171\nPreparing a low-rank posterior dose-minus-policy correction. Checking final-layer activation/layout and available physical directions before any weight edit.\n")
```