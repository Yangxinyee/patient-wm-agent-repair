Corrected numbers:

| | Policy decode from context (R²) | Residual sd | Same model, fully fitted (sd) | Dose decode from u (R²) |
|---|---|---|---|---|
| Insulin | 0.57 | 0.65 U/h | 0.35–0.49 | 0.998 |
| NE | 0.47 | 0.24 | 0.07–0.09 | 0.996 |

The dose itself is recoverable from u almost exactly. The expected dose is only partly recoverable from the model's state. For NE most of the policy runs through the previous hour's rate, which the model's state has forgotten.

So a residual computed inside the model, r = dose − decode(context), is mostly real residual for insulin. For NE it is mostly policy part, which the model has already priced in. Any NE gain κ therefore costs factual NLL of roughly κ²·Var(r)/(2σ²) on the MAP readings that follow an NE bundle.

This is an information limit, not a bug. I have about 0.0065 nats of budget: the 0.005 tolerance plus the 0.0015 gain from the z_head edit. κ_N will have to be calibrated against that budget, not just against the readout.

Next I need the helpers that compute u for edited doses (`u_rate`, `edit_atom`, `u_for`). With them I can build the PCA copy basis and the dose/presence decoders over the full edited range:
- insulin ×0–×3, up to 15 U/h;
- NE up to 3, beyond the recorded maximum of 1.0.

This run just prints their signatures and short sources.

```python
for f in [u_rate, find_atom]:
    try: print(f.__name__, inspect.signature(f))
    except Exception as ex: print(f, ex)
src=open('/work/wm2_rollout.py').read()
for nm in ['def edit_atom','def u_for']:
    i=src.find(nm); print(src[i:i+600]); print('----')
print('grid', np.round(grid[[0,1,-2,-1]],3), 'a_ins keys', list(a_ins.keys()) if isinstance(a_ins,dict) else type(a_ins))
```