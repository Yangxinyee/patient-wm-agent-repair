The records have NE as `event` rows with `rate` and `end_hour`, and `true_shift_ne` reproduces per-stay-hour targets. Now I will split each ne_x2 target into the part coming from NE records whose doubled rate stays at or below 1.0 and the part from records pushed above 1.0. Regressing the model's shift on both parts measures how much of each the edited model delivers.

Expected outcomes:
- In-range coefficient similar to the half edit's and out-of-range coefficient near 0: the out-of-range encoding is the culprit, and the fix belongs in encode_action.mlp1.
- Both coefficients low: the head channel itself under-delivers, and it is a gain problem.

I first check that my lag convention reproduces `true_shift_ne`.

```python
print(TU.event.unique()[:10])
ne=TU[TU.event.str.contains('norep',case=False)]; print(ne.head(3).to_string())
R={s:(g.hour.values.astype(float),g.rate.values.astype(float)) for s,g in ne.groupby('stay_id')}
G=np.array(gam)
def dec(s,h,fac,lagf):
    if s not in R: return 0.,0.
    t,r=R[s]; m=t>=12-1e-6; t,r=t[m],r[m]; k=lagf(h,t); ok=(k>=1)&(k<len(G)); g=np.where(ok,G[np.clip(k,0,len(G)-1)],0)*(fac-1)*r
    o=(fac*r>1+1e-9); return float((g*~o).sum()), float((g*o).sum())
E['tgt']=[true_shift_ne(s,h,2.0,gam) for s,h in zip(E.stay_id,E.hour)]
for nm,lf in [('h-floor',lambda h,t:(h-np.floor(t+1e-6)).astype(int)),('h-ceil',lambda h,t:(h-np.ceil(t-1e-6)).astype(int)),('h-floor-1',lambda h,t:(h-np.floor(t+1e-6)-1).astype(int))]:
    d=np.array([dec(s,h,2.0,lf) for s,h in zip(E.stay_id,E.hour)]); print(nm, np.abs(d.sum(1)-E.tgt.values).max().round(4))
```