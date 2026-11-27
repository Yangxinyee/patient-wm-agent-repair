With all four batches in, the tool has run the final edit on all 400 tuning stays. I pooled the batch results by hand, weighting by stay-hours:

| Readout | Tool | Truth (same rows) | Offline replica | Original model |
|---|---|---|---|---|
| insulin ×2 | −13.52 | −13.41 | −13.76 | about −1.8 |
| insulin ×½ | +6.49 | +6.71 | +6.67 | about +0.9 |
| norepinephrine ×2 | +4.35 | +4.47 | +4.39 | about −0.1 to −0.6 |
| norepinephrine ×½ | −1.29 | −2.24 | −1.21 | about −0.5 |

The stay-hour counts (2,009 for insulin, 1,080 for norepinephrine) match the rows my truth was computed on.

So the tool agrees with the offline copy within 1–3% for insulin and within about 1% for norepinephrine ×2:
- **Insulin ×2:** within 1% of the truth.
- **Norepinephrine ×2:** 3% low.
- **Insulin ×½:** 3% low.
- **Norepinephrine ×½:** only partly repaired, at 58% of the truth, deliberately limited by the factual cost.

This run does three things:
1. **Checks the comparison per batch:** it computes the truth on each batch's exact rows and a stay-bootstrap standard error for the tool-minus-truth gap.
2. **Nudges θ₂:** the shortfall on norepinephrine ×2 can be closed by raising θ₂ by the measured slope of 3.13, at essentially no factual cost. I apply this only if the gap exceeds 0.05.
3. **Records everything in the notebook:** this protects the results against a conversation restart.

What would confirm or refute the plan:
- **Confirm:** the gaps fall within about 1–2 standard errors. Then the calibration is as good as these data allow, and only the small θ₂ nudge is warranted.
- **Refute:** a gap of several standard errors would mean a real calibration error, which I would correct instead.

```python
import pandas as pd, numpy as np, torch, time, traceback as tb
T0=time.time()
try:
    rows=[]
    for k in [7,8,9,10]:
        d=pd.read_csv(f'/work/fwd_{k}/dose_edit_edited.csv'); d=d[d.reached].copy()
        d['truth']=[true_shift(int(s),int(t),e) for s,t,e in zip(d.stay_id,d.hour,d.edit)]; d['batch']=k; rows.append(d)
    A=pd.concat(rows); A['diff']=A['shift']-A['truth']
    print(A.groupby(['edit','batch'])[['shift','truth']].mean().round(2).unstack('batch').to_string())
    tot=A.groupby('edit')[['shift','truth']].mean().round(3); tot['n']=A.groupby('edit').size()
    rng=np.random.default_rng(0); se={}
    for ed,g in A.groupby('edit'):
        ps=g.groupby('stay_id')['diff'].agg(['sum','count']); bs=[]
        for b in range(2000):
            s=rng.integers(0,len(ps),len(ps)); bs.append(ps['sum'].values[s].sum()/ps['count'].values[s].sum())
        se[ed]=round(float(np.std(bs)),3)
    tot['se_diff']=pd.Series(se); print(tot.to_string())
    gap=float(tot.loc['ne_x2','truth']-tot.loc['ne_x2','shift']); th2n=0.83+gap/3.13 if abs(gap)>0.05 else 0.83
    P_,_,_=build_ne2(Pcur,th2n,0.5)
    cst=float(torch.as_tensor(tf_eval(P_,ds1,False)[0]).double().mean())-float(n0.mean())
    for n in ['attr_heads.value_bin.0.weight','attr_heads.value_bin.0.bias','attr_heads.value_bin.2.weight','attr_heads.value_bin.2.bias']: save_edit(n,P_[n])
    print('SAVED th2',round(th2n,3),'thh 0.5; offline tuning400 cost',round(cst,5),'edited',len(edited()))
    with open('/work/notes.md','a') as f:
        f.write(f'- Run293: TOOL dose_edit on ALL 400 tuning (fwd7-10, edit th2=.83 thh=.5, f=1.16): pooled tool vs truth: '+'; '.join(f"{e} {tot.loc[e,'shift']} vs {tot.loc[e,'truth']} (se {tot.loc[e,'se_diff']})" for e in tot.index)+f'. Then th2 -> {th2n:.3f} (gap {gap:.2f}/3.13), offline tuning cost {cst:.5f}; SAVED. Remaining: tool factual check of final edit on 400 tuning (4 passes), then FINAL ANSWER.\n')
except Exception: print('err',tb.format_exc()[-900:])
print('sec',round(time.time()-T0))
```