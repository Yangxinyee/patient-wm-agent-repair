The mismatch sits entirely in hours ≤ 12, the bundles copied from stored tool states: 10–17 mg/dL there, against 1.4 after hour 12, where run_cf recomputes. The bundle counts agree (5,116 each) and the copied `out` equals the stored one exactly, so the head is fine. The stored states themselves don't produce the right predictions in my head replica: either the stored h slots are in a different layout or stage, or the indexing is off for early bundles.

Next I recompute a stay from minute 0 with run_cf and compare its head predictions with the tool's at early hours, and its states with the stored ones.
- If the recomputed predictions match the tool (error about 1 mg/dL), I simply run everything from scratch and never use stored states.
- If they still miss, the problem is in the early-bundle indexing.

```python
s=2020000000; rt=RT[RT.stay_id==s].sort_values('b').reset_index(drop=True)
eps=torch.tensor(np.random.default_rng(int(s)%1000003).standard_normal((len(rt),4,32)))
Xo=X_of(s)
for k0m in (0,360,720):
    try:
        O,H,k0=run_cf(s,{},k0m,Xs); q,wc,P,mn=heads_on_states(O,H,s,rt,eps); mn=np.asarray(mn,float)
        x=rt.assign(mn=mn).merge(fc,on=['stay_id','event'],how='left',suffixes=('','_t'))
        x=x[np.round(x.minute/60)==x.hour]; e=(x.mn-x.predicted_mean_t if 'predicted_mean_t' in x else x.mn-x.predicted_mean).abs()
        hh=np.round(x.minute/60)
        print('from',k0m,'k0',k0,'err h<=12 %.2f h>12 %.2f'%(e[hh<=12].mean(),e[hh>12].mean()),'| H vs stored @k0-1 %.3f'%float((H[max(k0-1,0)]-torch.tensor(Xo[max(k0-1,0),192:960]).reshape(4,192)).abs().max()))
    except Exception as ex: print('from',k0m,'ERR',str(ex)[:150])
```