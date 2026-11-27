The output was cut because `m1`'s default arrays were printed, but the key facts came through:

- **`build_enc(beta)`.** It rebuilds the four repurposed `mlp1` units from fixed gates (`wGf`, `wZ`) and solves only their output weights by least squares (`lstsq`) against a target that is affine in beta. So scaling beta should scale the out-of-range correction linearly.
- **NE effect estimate.** It is a linear SNMM per unit rate (gam_ne lags 1–4: 3.16, 2.63, 1.78, 0.95). The NE policy caps rates at 1.0. The +4.85 ne_x2 target therefore rests on extrapolating that linear effect beyond the cap; I should test this later for curvature within the recorded range.

E5 at beta=1 added only +0.32. Next I need to know whether a bigger beta can deliver about 4 mmHg more, or whether the layer norm in the encoder saturates the representation.

This run:
- computes how far u moves, as a function of beta, for an ideal bin-16 atom at r′ = 1.5 and 2.0;
- rebuilds the encoder edit at beta=8, checking first that `build_enc(1)` reproduces the saved E5;
- saves the beta=8 version as the current edit for a dose_edit test.

If u grows roughly linearly with beta, a larger beta is a viable route. If it plateaus early, this route is capped.

```python
import numpy as np, torch
print({k: getattr(globals().get(k),'shape',None) for k in ['m10','dm','rp','zp_','Mo','act','wGf','wZ','NOR_U','MA_U']})
u10 = np.asarray(u_from_m(m10)).ravel()
for rpv in [1.5, 2.0]:
    print('r', rpv, [round(float(np.linalg.norm(np.asarray(u_from_m(m10 + B*(rpv-1)*dm)).ravel()-u10)),2) for B in [0.5,1,2,4,8,12,16,24]])
R1 = build_enc(1.0); R8 = build_enc(8.0)
print(type(R8))
it8 = list(R8.items()) if isinstance(R8, dict) else list(enumerate(R8))
it1 = list(R1.items()) if isinstance(R1, dict) else list(enumerate(R1))
nm = {(768,192):'encode_action.mlp1.0.weight',(768,):'encode_action.mlp1.0.bias',(192,768):'encode_action.mlp1.2.weight',(192,):'encode_action.mlp1.2.bias'}
new = {}; ok = True
for (k8,v8),(k1,v1_) in zip(it8,it1):
    a8 = np.asarray(v8, dtype=np.float32); a1 = np.asarray(v1_, dtype=np.float32)
    name = k8 if isinstance(k8,str) else nm[a8.shape]
    cur = np.asarray(load(name), dtype=np.float32)
    e1 = float(np.abs(a1-cur).max()); ok &= e1 < 1e-4
    print(name, a8.shape, 'R1-saved', round(e1,6), 'R8-R1', round(float(np.abs(a8-a1).max()),3))
    new[name] = a8
if ok:
    for n_, a_ in new.items():
        try: save_edit(n_, a_)
        except Exception: save_edit(n_, torch.from_numpy(a_))
    np.savez('/work/E5b8.npz', **new)
    print('SAVED beta=8', edited())
with open('/work/notes.md','a') as f:
    f.write('- Fwd13 (sel1, E5 beta=1): ins_x2 -13.91, ins_half 7.08, ne_x2 1.35 (E4 1.03, tgt 4.85), ne_half -2.16. E5 helps only +.32.\n- build_enc(beta): fixed gates, lstsq output weights; target affine in beta. Run279: trying beta=8 (/work/E5b8.npz).\n')
```