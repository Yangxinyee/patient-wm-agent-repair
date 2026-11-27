The core runs in float32, so the head code in head2.py must use float64 weights. I cast the wrong way last time. I'll pass the view context as float64 and rerun the same z-token response test.

```python
def next_pred(k, ztok_add=None, codes=('glucose_finger','map_arterial')):
    c, z, h, o = step_local(k, ztok_add=ztok_add) if ztok_add is not None else step_local(k)
    dt1 = dts[k+1]; hf = h*torch.exp(-dt1*torch.exp(fa)); cv = ctx_from(o, hf, dt1).double()
    out = []
    for cd in codes:
        lv = LV[cd][k+1]; la = (INFO[k+1]['t']-LT[cd][k+1])*60.0
        r = value_mean(cv, [cd], lv, la); out.append(float(np.asarray(r).ravel()[0]))
    return np.array(out)
k = ks_ins[5]; b = next_pred(k); print('base', b.round(2), 'actual', INFO[k+1].get('glucose_finger'), INFO[k+1].get('map_arterial'))
for s in [1,3,10,30]: print(s, (next_pred(k, ztok_add=s*dv) - b).round(3))
```