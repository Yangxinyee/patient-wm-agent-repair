For stay …024 the direct replica is also wrong before hour 12 (49 mg/dL) but right after it (1.8). It agrees exactly with M, so the stored chunk is faithful to what the replica produced. The problem is the stored tool states used for the early bundles: for most stays they don't match what the tool's heads saw. The bundles recomputed from hour 12 on are fine, which suggests the state forgets a bad start quickly.

Next I recompute this stay entirely from scratch (run_cf from a very negative minute, so no stored states are used) and check the early-hour error.
- If it drops to about 1.5, I rebuild the factual baselines and counterfactual variants from scratch for all stays and drop the stored states.
- If run_cf can't start from bundle 0, I seed it differently.

```python
for k0m in (-100000,-90):
    try:
        O,H,k0=run_cf(s,{},k0m,Xs); q,wc,P,mn=heads_on_states(O,H,s,rt,eps); mn=np.asarray(mn,float)
        e=np.abs(mn-x[pmc].values); hh=x.hour.values
        print('from',k0m,'k0',k0,'err h<=12 %.2f h>12 %.2f'%(e[hh<=12].mean(),e[hh>12].mean()))
    except Exception as ex: print('from',k0m,'ERR',str(ex)[:200])
Xo=X_of(s); print('stored out norms first 4', np.round(np.linalg.norm(np.asarray(Xo[:4,:192]),axis=1),2), '| sid order ok', bool((S['stay_id'][np.where(S['stay_id']==s)[0]]==s).all()), 'n', int((S['stay_id']==s).sum()), len(BUN[s]))
```