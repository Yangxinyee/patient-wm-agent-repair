A_ holds 27,963 head-input rows across datasets 0-4, with code, drug flags, the dev targets and stay. The per-dataset row tables have stay and hour. Next I'll attach the recorded NE rate at lags 1-3 to each row from the training grid. Then I'll test with stay-grouped cross-validated ridge how well the most recent NE rate level can be read from the value head's input on map_arterial rows, compared with the deviation target nedev1.

A clearly higher R2 for the level would support switching to a level-based NE correction; a similarly low one would not.

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
RRall = pd.concat([datasets[i][1] for i in range(5)], ignore_index=True)
print(len(RRall), (RRall.stay.values == A_.stay.values).mean())
neG = Gtr['ne']
def lagrate(G, stays, hours, k):
    out = np.zeros(len(stays))
    for i,(s,h) in enumerate(zip(stays, hours)):
        c = int(h)-k
        if s in G.index and c in G.columns:
            v = G.at[s,c]; out[i] = 0.0 if pd.isna(v) else v
    return out
for k in (1,2,3):
    RRall[f'rne{k}'] = lagrate(neG, RRall.stay.values, RRall.hour.values, k)
WCm = WCall.mean(0)
m = (A_.code.values==7)
def cvr2_ridge(X, y, g, lam=1.0):
    pr = np.zeros_like(y)
    for tr,te in GroupKFold(5).split(X,y,g):
        pr[te] = Ridge(alpha=lam).fit(X[tr],y[tr]).predict(X[te])
    return 1-((y-pr)**2).sum()/((y-y.mean())**2).sum()
for tgt in ['rne1','rne2','nedev1']:
    y = (RRall[tgt] if tgt in RRall else A_[tgt]).values[m].astype(float)
    y = np.nan_to_num(y)
    print(tgt, 'n', m.sum(), 'sd %.3f'%y.std(), 'R2', [round(cvr2_ridge(WCm[m], y, A_.stay.values[m], l),3) for l in (0.1,1,10)])
```