`step_err(k)` returns three things: the predicted h (4×192), the predicted out (192), and a float. The float for bundle 5 is 0.00096, which looks like the relative error against the tool's state. My own comparison gave NaN, probably because HHt holds NaNs for some bundles (for example the first bundle of each stay) or uses a different index.

I will collect that returned float over many bundles and summarise it. Median errors of about 1e-3 with a modest maximum mean the local step is faithful and I can simulate edits on the CPU. A heavy tail would point to particular bundle types that are mishandled, such as first bundles or drug-only bundles. I also count NaNs in HHt to explain my failed comparison.

```python
import numpy as np, torch
ee=[]; kk_=[]
for k_ in range(1, 5156, 7):
    try:
        r_=step_err(k_); ee.append(r_[2]); kk_.append(k_)
    except Exception as ex:
        pass
ee=np.array(ee,float)
print("n", len(ee), "nan", np.isnan(ee).sum())
f_=ee[~np.isnan(ee)]
print("err median %.3g  90%% %.3g  99%% %.3g max %.3g"%(np.median(f_),np.percentile(f_,90),np.percentile(f_,99),f_.max()))
print("HHt nan rows", int(torch.isnan(torch.as_tensor(HHt)).any(-1).any(-1).sum()))
print("worst ks", [kk_[i] for i in np.argsort(-np.nan_to_num(ee))[:10]], np.sort(f_)[-10:])
```