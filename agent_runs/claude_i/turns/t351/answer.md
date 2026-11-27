The factual damage has a clear pattern: it comes almost entirely from the first hours of each stay.
- In hours −1 to 5, the edit shifts predicted glucose by −6.4 on average and MAP by +1.3.
- After hour 5, the shifts are small (−0.2 to −0.3 for glucose, about +0.06 for MAP).
- Where the edit barely moves the prediction, the NLL change is about 0.
- Where it moves it a lot, NLL is much worse (+0.09 for finger glucose, +0.52 for arterial MAP).

My working hypothesis is that the dose-deviation channel misfires early in a stay. Either the gate fires wrongly there, or the expected-dose estimate extrapolates badly from the unusual early context, so it injects large spurious values.

I also still need the edited model's dose response from pass 5 (that part of the output was cut off). And I need the stay ids for the next forward requests.

This run will show:
- the edited dose response from pass 5;
- the factual NLL change and the error-versus-shift slope by hour band;
- whether the early damage is concentrated in a few stays;
- the stay ids.

The hypothesis is confirmed if the damage is confined to early hours while the later hours are neutral (slope near or above 0). It is refuted if the later hours also show harm or a negative slope.

```python
import json, numpy as np, pandas as pd
S=json.load(open('/work/fwd_5/summary.json'))['results']
for m in S: print(m, {k:v['mean_shift'] for k,v in S[m]['edits'].items()})
de=pd.read_csv('/work/fwd_5/dose_edit_edited.csv'); r=de[de.reached.astype(bool)]
print(r.groupby(['edit', pd.cut(r.hour,[12,16,20,25])])['shift'].mean().round(2).unstack().to_string())
fo=pd.read_csv('/work/fwd_6/factual_original.csv'); fe=pd.read_csv('/work/fwd_6/factual_edited.csv')
dm=fe.predicted_mean-fo.predicted_mean; dn=fe.nll-fo.nll; res=fo.observed-fo.predicted_mean
q=pd.cut(fo.hour,[-2,1,4,7,12,18,25])
for ev in ['glucose_finger','map_arterial','map_cuff']:
    i=fo.event==ev
    t=pd.DataFrame({'dn':dn[i],'dm':dm[i],'r':res[i],'q':q[i]})
    g=t.groupby('q',observed=True).apply(lambda x: pd.Series({'n':len(x),'dNLL_sum':x.dn.sum(),'mdm':x.dm.mean(),'madm':x.dm.abs().mean(),'b':(x.dm*x.r).sum()/max((x.dm**2).sum(),1e-9)}))
    print(ev); print(g.round(3).to_string())
ps=dn.groupby(fo.stay_id).sum().sort_values(ascending=False)
print('total dNLL sum',round(dn.sum(),1),'top5 stays',ps.head(5).round(1).to_dict(),'top10 share',round(ps.head(10).sum()/dn.sum(),2))
early=fo.hour<=4
print('early rows share of dNLL', round(dn[early].sum()/dn.sum(),2))
ids=sorted(fo.stay_id.unique()); print('IDS', ','.join(map(str,ids)))
with open('/work/notes.md','a') as f:
    f.write('- run291: fwd5 edited (full-strength edit) shifts: '+str({k:v['mean_shift'] for k,v in S.get('edited',{}).get('edits',{}).items()})+'; factual damage concentrated early hours (dm gf -6.4 in h<=5, ~-0.25 later). Hypothesis: dev channel misfires early. Current saved edit: kv at 0.5x (DW_full in /work/DW_kv_full.npy).\n')
```