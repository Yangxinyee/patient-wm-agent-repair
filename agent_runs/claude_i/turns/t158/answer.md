`step_err(5)` worked, but it returns tensors rather than a scalar error, and printing them filled the output. The slots it returns have entries up to about ±150, and the four slots are almost identical to one another.

Next I will print only the types and shapes of what `step_err` returns. I will also compute the relative error of each returned 4×192 tensor against the tool's recorded state for that bundle (HHt[k]). I do this for about 300 bundles and summarise the median, 90th percentile and maximum. A median relative error below about 1e-3 means my local core step reproduces the model and can be used to test edits. A larger error means I must find which part of the step is mis-specified.

```python
import numpy as np, torch
r0=step_err(5)
print(type(r0), len(r0) if isinstance(r0,(tuple,list)) else '')
for i_,x_ in enumerate(r0 if isinstance(r0,(tuple,list)) else [r0]):
    print(i_, type(x_).__name__, tuple(x_.shape) if hasattr(x_,'shape') else x_)
rel=[];relo=[]
for k_ in range(2, 300):
    try:
        r_=step_err(k_)
    except Exception as ex:
        continue
    cand=[x for x in (r_ if isinstance(r_,(tuple,list)) else [r_]) if hasattr(x,'shape')]
    tgtH=torch.as_tensor(HHt[k_]).double()
    errs_=[]
    for x_ in cand:
        x_=torch.as_tensor(x_).double()
        if tuple(x_.shape)==tuple(tgtH.shape):
            errs_.append(float((x_-tgtH).norm()/tgtH.norm()))
    if errs_: rel.append(min(errs_))
rel=np.array(rel)
print("n", len(rel), "rel err median %.3g 90%% %.3g max %.3g"%(np.median(rel), np.percentile(rel,90), rel.max()))
```