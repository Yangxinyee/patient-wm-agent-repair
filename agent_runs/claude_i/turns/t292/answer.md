Run 254 showed three things.
- **Drug arrays:** `Fi` and `Fn` hold per-bundle drug rates: column 0 is the current hour's rate, columns 1–5 the previous hours, and column 6 is my kernel-implied effect. That gives ready-made targets for the deviation feature.
- **Record layout:** columns are `stay_id, hour, event, value, rate, ...`.
- **`run_stay_z`:** it needs torch tensors.

I'll call `run_stay_z` with tensors and check that its states for one stay match the tool's stored states. If they match, I can run whole stays with an added z signal and measure how the next glucose and MAP predictions respond, all locally without forward passes.

```python
import torch
Et = torch.from_numpy(np.asarray(Eall, np.float32)); Ut = torch.from_numpy(np.asarray(Uall, np.float32))
out = run_stay_z(ks0, Et, Ut)
print(type(out), len(out) if isinstance(out,(tuple,list,dict)) else '', [ (getattr(x,'shape',None), type(x).__name__) for x in (out if isinstance(out,(tuple,list)) else (out.values() if isinstance(out,dict) else [out]))])
o = out[0] if isinstance(out,(tuple,list)) else out
try:
    on = N(o); print('shape', on.shape, 'err vs X3 out', np.abs(on[:, :192]-X3n[ks0][:, :192]).max() if on.shape[-1]>=192 else '')
except Exception as ex: print('cmp fail', str(ex)[:80])
```