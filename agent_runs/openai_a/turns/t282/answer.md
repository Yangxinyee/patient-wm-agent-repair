The full-cohort average passes, but the stay-bootstrap upper bound is 0.00557, so the safety margin is not robust. I’ll inspect the gate under recorded, doubled, and halved insulin before choosing a lower-risk adjustment. A systematic gate drop under halving would support a dose-invariance fix; stable gate scores would point instead to the learned contrast.

```python
from scipy.special import erf
def gelu_np_recovery(x):
    return .5*x*(1+erf(x/np.sqrt(2.0)))
cfg = activity_gate_config
base_gw = sc_gate_w + cfg["finger_gate_strength"]*selective_sep_w
base_gb = sc_gate_b + cfg["finger_gate_strength"]*selective_sep_b
print("=== SYNTHETIC GATE DIAGNOSTICS ===")
rows = []
for mult,c in sc_syn.items():
    x = c["x"]
    a = x@cfg["w"] + cfg["b"]
    bg = x@base_gw + base_gb
    gate = bg + cfg["activity_strength"]*(a-cfg["activity_centre"])
    r = x@cfg["contrast_w"] + cfg["contrast_b"]
    correction = cfg["scale"]*(gelu_np_recovery(gate+r)-gelu_np_recovery(gate))
    rows.append({"mult":mult,"activity_mean":a.mean(),"activity_min":a.min(),
                 "activity_max":a.max(),"base_gate_mean":bg.mean(),
                 "gate_mean":gate.mean(),"gate_positive_fraction":(gate>0).mean(),
                 "residual_mean":r.mean(),"correction_mean":correction.mean(),
                 "correction_sd":correction.std()})
print(pd.DataFrame(rows).to_string(index=False))
print("meta columns",list(sc_syn[1.0]["meta"].columns))
print(sc_syn[1.0]["meta"].head(3).to_string(index=False))
print("\n=== EVALUATION HELPERS ===")
for fnname in ["screen_value_candidate","cache_losses","sc_evaluate","selective_cache"]:
    fnobj = globals()[fnname]
    print(fnname,inspect.signature(fnobj),
          "globals",fnobj.__code__.co_names,
          "locals",fnobj.__code__.co_varnames)
print("\n=== BASELINE TENSOR IDENTITY ===")
for n,v in sc_original.items():
    orig = load_original(n).numpy()
    print(n,np.asarray(v).shape,"max diff from original",float(np.max(np.abs(np.asarray(v)-orig))))
print("sc_rank",sc_rank)
print("bias_comp norm",np.linalg.norm(sc_bias_comp))
print("\n=== RELEVANT FIT ARRAYS ===")
for k,v in sorted(dict(globals()).items()):
    if not callable(v) and hasattr(v,"shape") and any(s in k.lower() for s in
            ["invar","active","activity","r233","r234","r235","r236","r237","gate_fit"]):
        print(k,type(v).__name__,tuple(v.shape))
```
