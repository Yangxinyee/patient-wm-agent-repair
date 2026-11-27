The call failed on a dtype mismatch: the core reproduction (WT) runs in float64, while the head code in head2.py uses float32 weights. I'll cast the view context to float32 before calling `value_mean` and rerun the same test. The test is unchanged: does the next glucose prediction respond strongly and smoothly to z-token additions?

```python
def next_pred(k, ztok_add=None, codes=('glucose_finger','map_arterial')):
    c, z, h, o = step_local(k, ztok_add=ztok_add) if ztok_add is not None else step_local(k)
    dt1 = dts[k+1]; hf = h*torch.exp(-dt1*torch.exp(fa)); cv = ctx_from(o, hf, dt1).float()
    out = []
    for cd in codes:
        lv = LV[cd][k+1]; la = (INFO[k+1]['t']-LT[cd][k+1])*60.0
        r = value_mean(cv, [cd], lv, la); out.append(float(np.asarray(r).ravel()[0]))
    return np.array(out)
print('dtypes', fa.dtype, dts.dtype)
k = ks_ins[5]; b = next_pred(k); print('base', b.round(2))
for s in [1,3,10,30]: print(s, (next_pred(k, ztok_add=s*dv) - b).round(3))
```