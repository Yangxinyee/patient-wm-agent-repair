Run 81 recovered my local head code. The `Head` class computes ctx (time-flowed h, out_norm, gap), the z prior, `with_code` (query_norm view, code embedding, last_numeric, last value bin), the hidden GELU layer and the logits. The kernel also holds the states SX (5301×992, from fwd_2) and per-reading arrays for the 1865 glucose_finger readings in fwd_3: FH, DH and PMm.

Run 77 left a median error of 2.46 mg/dL against the model's predicted means. That could be a bug in my code, or just the model's 4 random prior draws of z. I need to know which before fitting probes on hidden activations. First I'll look at what DH holds and how the errors are distributed. Next I'll compare against a many-draw average and the expected spread of a 4-draw average.

```python
print(DH.columns.tolist()); print(DH.head(4).to_string())
print(type(CEN), (list(CEN.keys())[:6] if isinstance(CEN, dict) else np.shape(CEN)))
pm_loc = np.array([float(x) for x in PMm])
mcol = [c for c in DH.columns if 'pred' in c.lower()]
print('pred cols', mcol)
if mcol:
    err = pm_loc - DH[mcol[0]].values
    print('err quantiles', np.round(np.quantile(err, [.05,.25,.5,.75,.95]),2), 'mean', err.mean().round(2), 'mad', np.median(np.abs(err)).round(2))
    for c in DH.columns:
        if c not in mcol and pd.api.types.is_numeric_dtype(DH[c]) and DH[c].nunique()>3:
            print(c, 'corr with |err|', np.corrcoef(np.abs(err), DH[c].values)[0,1].round(3))
```