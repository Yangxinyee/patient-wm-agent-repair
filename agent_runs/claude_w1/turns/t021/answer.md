A norepinephrine innovation of +1 mcg/kg/min at hour t raises arterial MAP by:

| Hours after t | 1 | 2 | 3 | 4 | 6 | 12 |
|---|---|---|---|---|---|---|
| Total MAP rise (mmHg) | 3.4 | 5.1 | 5.4 | 5.1 | 3.8 | 1.0 |

The innovation persists in later rates through titration: 0.74 of it remains after 1 hour, 0.57 after 2, 0.45 after 3. So part of the total effect comes from the extra dose carried forward.

A rough deconvolution gives a fixed-dose effect per hour of infusion of about +3.4, +2.6, +1.6, +0.9 mmHg at 1–4 hours, decaying. So norepinephrine raises MAP here, and the effect fades over several hours.

**Model-side issue.** Doubling pushes norepinephrine rates above the 1.0 cap. Those rates fall in rate/dose bins 13–16, which never occur in training, so their embeddings are untrained.

Next I'll finish the identification checks in one run:
- whether the norepinephrine innovation is random (no relation to past innovations or other history);
- cross-effects: norepinephrine on glucose, insulin on MAP;
- whether the norepinephrine effect depends on the rate level (saturation).

After that, the first forward pass, to see what the model currently predicts.

```python
# NE innovation: residual of log(rate) given cell (prev-rate bin x band), excluding cap
T['lr'] = np.log(T.norepinephrine/T.ne_prev)
T['lr_res'] = T.lr - T.groupby('cell').lr.transform('mean')
g2 = T.groupby('stay_id')
T['lr_res_prev'] = g2.lr_res.shift(1)
cc = T[['lr_res','lr_res_prev','m_now','g_now','insulin','h']].corr()['lr_res']
print("NE innovation correlations:\n", cc.to_string())
# Cross effects: NE innovation -> glucose; insulin shock -> MAP
res=[]
for k in [1,2,3,4,6]:
    mk = T[f'G+{k}'].notna() & T.g_now.notna()
    sub=T[mk]; C=cells[mk].values
    X=np.column_stack([C, sub.g_now, sub.ne_prev, sub.norepinephrine])
    r,_=ols(sub[f'G+{k}'],X); 
    sub2 = S[S[f'MA+{k}'].notna() & S.map_arterial.notna()]
    X2=np.column_stack([np.ones(len(sub2)), sub2.g_now, sub2.map_arterial, sub2.shock]+[(sub2.ins_base==b).astype(float) for b in [2,3,4,6]])
    r2,_=ols(sub2[f'MA+{k}'],X2)
    res.append((k, r.b.iloc[-1], r.se.iloc[-1], r2.b.iloc[3], r2.se.iloc[3]))
print(pd.DataFrame(res,columns=['k','NE->G','se','INS->MAP','se2']).to_string())
# NE effect by rate level (k=1,2)
for lo,hi in [(0,0.15),(0.15,0.35),(0.35,0.6),(0.6,0.9)]:
    mk = T['MA+1'].notna() & T.map_arterial.notna() & (T.ne_prev>lo)&(T.ne_prev<=hi)&(T.norepinephrine<1.0)
    sub=T[mk]; C=cells[mk].values
    X=np.column_stack([C, sub.map_arterial, sub.ne_prev, sub.norepinephrine])
    r,_=ols(sub['MA+1'],X)
    print(lo,hi,mk.sum(), r.b.iloc[-1], r.se.iloc[-1])
```