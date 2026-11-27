The rollout takes about 0.5 s per stay for 6 conditions × 20 paths, and the factor-1 sanity condition gives exactly zero shift. For stay 2020000000 at hour 16, my insulin x2 shift on glucose_finger is −0.301; the tool reported −0.311 (se 0.011). Now I'll check all 100 batch1 stays against fwd_1. I'll compare per-stay-hour shifts on the tool's reached rows for the four edits, both the pooled means (tool: −3.25, +1.77, +0.62, −0.82) and the row correlation. Pooled means within about 10% and high correlation would show that the replica rollout reproduces the tool's dose_edit readout, so I can design and score edits offline.

```python
EDN={'insulin_x2':((8,2.0),6),'insulin_half':((8,0.5),6),'ne_x2':((9,2.0),7),'ne_half':((9,0.5),7)}
conds4=[None,(8,2.0),(8,0.5),(9,2.0),(9,0.5)]
def run_batch(W, stays, npaths=20, seed=1):
    R={}
    for s in stays:
        R[s]=rollout(W, BUN[s], conds4, npaths=npaths, seed=seed)
    return R
t0=time.time(); R0=run_batch(W0, b1); print('time', round(time.time()-t0,1))
def compare(R, f1, label=''):
    out_=[]
    for j,(ed,(cd,oc)) in enumerate(EDN.items()):
        sub=f1[(f1.edit==ed)&(f1.reached)]
        mine=[]; tool=[]
        for r in sub.itertuples():
            key=(int(r.hour),oc)
            if key in R[r.stay_id]:
                a=R[r.stay_id][key]; mine.append(float((a[j+1]-a[0]).mean())); tool.append(r.shift)
        mine=np.array(mine); tool=np.array(tool)
        out_.append((ed, len(mine), round(mine.mean(),3), round(tool.mean(),3), round(np.corrcoef(mine,tool)[0,1],3)))
    return out_
for r_ in compare(R0, f1): print(r_)
```