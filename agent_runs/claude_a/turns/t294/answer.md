Each tuple holds the hour, the code, and a [specs × paths] tensor of drawn values. So a stay-hour's shift is the path-mean of (edited − reference) under common random numbers. A row counts as reached if it is after the drug's first run from hour 12 on (`I['insulin']`, `I['norepinephrine']`). I'll write a pooled evaluator matching the tool's readout:
- insulin edits: glucose_finger rows after the first insulin;
- norepinephrine edits: map_arterial rows after the first norepinephrine;
- truth from `true_shift`.

I'll run it on 40 tuning stays (sel1) for both the original model and the θ=−2 build, with all four edits. If my offline rollout matches the tool, the original should reproduce roughly −1.9/+1.0 for insulin and −0.4/−0.5 for norepinephrine, against a truth of about −14/+7 and +4/−2. The build's insulin shift should be substantially larger than the original's.

```python
EDN=['insulin_x2','insulin_half','norepinephrine_x2','norepinephrine_half']
SPECS=[{'ins':1.,'ne':1.,'shift':None}]+[{'ins':(mult[n] if n.startswith('ins') else 1.),'ne':(mult[n] if n.startswith('nor') else 1.),'shift':None} for n in EDN]
def eval_shifts(P, stays, n_paths=20, seed=0):
    acc={n:[[],[]] for n in EDN}
    for s_ in stays:
        L,I=rollout_gen(tu, s_, P, SPECS, n_paths, seed)
        for (hr,code,vals) in L:
            v=vals.numpy() if hasattr(vals,'numpy') else np.asarray(vals)
            for k,n in enumerate(EDN,1):
                drug='insulin' if n.startswith('ins') else 'norepinephrine'; cc='glucose_finger' if drug=='insulin' else 'map_arterial'
                if code==cc and drug in I and np.isfinite(I[drug]) and hr>I[drug]:
                    acc[n][0].append(float((v[k]-v[0]).mean())); acc[n][1].append(float(true_shift(s_,hr,n)))
    return {n:(np.mean(a[0]) if a[0] else np.nan, np.mean(a[1]) if a[1] else np.nan, len(a[0])) for n,a in acc.items()}
st40=S['sel1'][:40]; t=time.time()
rPO=eval_shifts(PO,st40); print('PO', {n:tuple(np.round(v[:2],2))+(v[2],) for n,v in rPO.items()}, round(time.time()-t),'s')
rPR=eval_shifts(PR,st40); print('PR', {n:tuple(np.round(v[:2],2))+(v[2],) for n,v in rPR.items()})
```