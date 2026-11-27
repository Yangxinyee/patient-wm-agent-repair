Each bundle is a pair (time in hours, list of atom dicts), and the rows of `X_` line up one-to-one with bundles from `row_start`.

Now I will check the head against the tool directly. For every reading in fwd_3 I take the tool's own state after the previous bundle and apply the time flow. I build the "last value of the same code" register and compute the value-head predicted mean, then compare it with the tool's `predicted_mean`.

For this first scan I fix:
- z at the prior mean (no sampling);
- the register layout as [has_last, log(1+age)/10, z_v, has_v, z_d, has_d, z_r, has_r];
- status row 0;
- the gap input as log(dt)/10.

I then scan all 15 `type_emb` rows. One type index should stand out with near-perfect correlation and a mean absolute error of a fraction of a mg/dL. If none does, the layout or gap assumption is wrong and I will vary those next.

```python
import torch, numpy as np
T = lambda n: torch.as_tensor(np.asarray(W64[n]), dtype=torch.float64)
gelu = torch.nn.functional.gelu
def lnorm(x, n): return torch.nn.functional.layer_norm(x, (x.shape[-1],), T(n+'.weight'), T(n+'.bias'), 1e-5)
def mlp2(x, n): return gelu(x @ T(n+'.0.weight').T + T(n+'.0.bias')) @ T(n+'.2.weight').T + T(n+'.2.bias')
# test helper conventions
try: print('zsc', zsc('glucose_finger','value',176.0), 'fbin', fbin('glucose_finger','value',176.0))
except Exception as ex: print('helper err', ex)
print('code_id', code_id)
recs=[]
for sid_ in selB:
    Bq = bundles_B[sid_]; r0 = row_start[sid_]
    last = {}
    for k,(t,ats) in enumerate(Bq):
        for a in ats:
            c = a['code']
            if k>0 and c in ('glucose_finger','glucose_lab','map_arterial','map_cuff') and 'value' in a:
                L = last.get(c)
                recs.append((sid_, t, c, a['value'], r0+k-1, (t-Bq[k-1][0])*3600.0,
                             0 if L is None else 1, 0.0 if L is None else (t-L[0])*3600.0, 0.0 if L is None else L[1]))
        for a in ats:
            if 'value' in a: last[a['code']] = (t, a['value'])
R_ = pd.DataFrame(recs, columns=['sid','t','code','obs','row','dt','hl','age','lv'])
key = f3.set_index(['stay_id','hour','event'])['predicted_mean']
R_['pm'] = [key.get((s_,round(t_,6),c_), np.nan) for s_,t_,c_ in zip(R_.sid,R_.t,R_.code)]
print(R_.shape, R_.pm.isna().sum())
```