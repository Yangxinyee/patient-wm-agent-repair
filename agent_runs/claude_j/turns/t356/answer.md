Forward pass 16 is a serious warning. On 100 tuning stays outside my test set, mk2(4.5, 15) costs +0.0100, double the limit. On the 75 test stays it cost only +0.0026.
- Both G events got worse: GF +0.0158 and GL +0.0086.
- Unlike on the test set, the M events also got worse: MA +0.0090 and MC +0.0064.

**What I conclude:**
- The 75 test stays are a selected subset (nearly all received drug A), and the edit is unusually cheap on them. The evaluation uses stays like the general tuning population, where this edit would fail the factual limit.
- I must shrink the edit substantially and validate it on the non-selected batches.

**Next step (Python run 298 of 300):**
- Save a much smaller fallback, mk2(2.2, 5). This scales both amplifications down to about 30% of their deviation from no edit. It should cost about 0.003 on batch 1 if the cost grows linearly, and about 0.001 if quadratically.
- Split the per-stay cost of forward passes 15 and 16 by whether a stay had drug A, drug B, both or neither. This tells me which channel causes the cost in the general population. I will use that to choose the final candidate with run 299, keeping run 300 to fall back to this one.

**How I will read the split:**
- If the cost sits in stays without drug B, or comes from the B channel, the final candidate should cut kB and keep more kA.
- If it is spread over stays with drug A, kA itself must stay small.

```python
import numpy as np, pandas as pd
mk2(2.2,5.0)
W1=np.asarray(load('core.prelude.0.attn.kv.weight'),dtype=np.float64); print('saved |D| %.3f'%np.linalg.norm(W1-W0))
R=['glucose_finger','glucose_lab','map_arterial','map_cuff']
try:
    ec=[c for c in tu.columns if tu[c].dtype==object and tu[c].isin(R).any()][0]
    oth=sorted(set(tu[ec].unique())-set(R)); print('ev col',ec,'other',oth)
    ea=[x for x in oth if 'ins' in x][0]; eb=[x for x in oth if 'nor' in x][0]
    cA=tu[tu[ec]==ea].groupby(tc).size(); cB=tu[tu[ec]==eb].groupby(tc).size()
except Exception as ex: print('E1',repr(ex)[:150])
def split(k):
    o=pd.read_csv('/work/fwd_%d/factual_original.csv'%k); d=pd.read_csv('/work/fwd_%d/factual_edited.csv'%k)
    key=['stay_id','hour','event','observed']
    if not (o[key].values==d[key].values).all():
        o=o.sort_values(key,kind='stable').reset_index(drop=True); d=d.sort_values(key,kind='stable').reset_index(drop=True)
    o['dn']=d['nll'].values-o['nll'].values; o['G']=o['event'].str.startswith('glu'); N=len(o)
    o['dG']=o['dn']*o['G']; o['dM']=o['dn']*(~o['G'])
    s=o.groupby('stay_id').agg(n=('dn','size'),dG=('dG','sum'),dM=('dM','sum'))
    s['A']=[cA.get(x,0)>0 for x in s.index]; s['B']=[cB.get(x,0)>0 for x in s.index]
    t=s.groupby(['A','B']).agg(ns=('n','size'),nr=('n','sum'),dG=('dG','sum'),dM=('dM','sum')); t[['dG','dM']]=t[['dG','dM']]/N
    tot=(s.dG+s.dM)/N; top=tot.sort_values(ascending=False)
    return t.round(5), round(tot.sum(),5), round(top.iloc[:5].sum(),5)
for k in (15,16):
    try: t,a,b=split(k); print(k,'total',a,'top5 stays',b); print(t.to_string())
    except Exception as ex: print('E2',k,repr(ex)[:150])
with open('/work/notes.md','a') as fh: fh.write("""- fwd15 mk2(4.5,15) sel1: +0.00256 (GF+.0103 GL+.0096 MA-.0029 MC-.0048). fwd16 SAME edit on 100 non-sel1 tuning stays (batch1): +0.0100 (GF+.0158 GL+.0086 MA+.0090 MC+.0064) -> FAILS. sel1 unrepresentative (cheap)!
- non-sel1 tuning batches (offsets from 2020000000, excluding sel1): b1 1..133, b2 134..255, b3 256..374, b4 375..399.
- Run298: fallback mk2(2.2,5) saved + per-stay cost split by drug presence (fwd15/16). Run299 = final candidate, run300 = reserve.
""")
```