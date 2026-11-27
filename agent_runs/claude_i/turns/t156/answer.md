The helper signatures show I did build a full local step in session 3. `core_step(hf, e, u, z, M, ages, V)` covers the prelude, blocks with cross-attention and age keys, and coda. `check` and `step_err` compare it with the tool's recorded states (HHt, OUTt, Ee, Uu, ZZt for 5,156 bundles of 100 stays). The value-head helpers (`pred_mean_s`, `pred_mean_v`) rebuild the predicted glucose mean from a state, and `view_ctx` rebuilds the head's view. I never recorded how accurate the step is.

So the next thing is to run `step_err` on many bundles and compare the error with the size of h. A median relative error around 1e-3 or less means I have a usable local simulator and can test edits on the CPU before spending forward passes. A large error means some detail of the step is wrong and must be fixed first.

```python
import numpy as np, torch
print("check sig default V; HHt", tuple(HHt.shape), "OUTt", tuple(OUTt.shape))
errs=[]; scal=[]
for k_ in range(1, 400):
    try:
        e_=step_err(k_)
        errs.append(float(e_) if not isinstance(e_, (tuple,list)) else [float(x) for x in e_])
        scal.append(float(torch.as_tensor(HHt[k_]).norm()))
    except Exception as ex:
        errs.append(None); scal.append(None)
ok_e=[(e,s) for e,s in zip(errs,scal) if e is not None]
print("n ok", len(ok_e), "first few", errs[:8])
ee=np.array([e if np.isscalar(e) else e[0] for e,s in ok_e]); ss=np.array([s for e,s in ok_e])
print("err median %.4g  90pct %.4g max %.4g | h-norm median %.3g"%(np.median(ee),np.percentile(ee,90),ee.max(),np.median(ss)))
print("notes var len", len(notes)); print(notes[-1500:])
```