No setting reproduces the old helper's probabilities exactly (the best differs by 0.03), so it probably used its own random z draws. I am spending too many runs on this, so I will look for a simpler explanation of my 4 mg/dL error.

The 17th class, bin 0, means "no value". If the tool computes the mean as the sum over bins 1–16 of p(b) × centre without renormalising, it comes out lower than mine whenever p(no value) is above zero. Even p ≈ 0.02 would shift it by about 3.5 mg/dL, about the error I see.

Next I will compare the signed error and the error with and without renormalisation, using my all-GELU, type-1, layout-A head with sampled z.
- **If the unrenormalised mean is close to the tool (error well below 2):** my head is right, and only the readout formula differed.

```python
def pm_both(r_, ns=32):
    pm_r, p_ = pred_mean_s(r_, ns=ns)
    ev_ = r_[1]
    return pm_r, float((p_[1:] * CENT[ev_]).sum()), float(p_[0])
vals_ = np.array([pm_both(r_) for r_ in sub_r]); tool_ = np.array([r_[3] for r_ in sub_r])
print('renorm: MAD', round(np.mean(np.abs(vals_[:,0]-tool_)),3), 'signed', round(np.mean(vals_[:,0]-tool_),3))
print('no renorm: MAD', round(np.mean(np.abs(vals_[:,1]-tool_)),3), 'signed', round(np.mean(vals_[:,1]-tool_),3))
print('p(no value) mean', round(vals_[:,2].mean(),4), 'max', round(vals_[:,2].max(),4))
```