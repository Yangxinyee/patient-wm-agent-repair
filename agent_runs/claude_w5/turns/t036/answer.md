The bundle structure is now clear:
- Readings form one bundle on the hour; drug records form a separate bundle at hour + 5 min.
- From the second record on, a drug bundle holds 4 atoms: the END of the previous insulin and norepinephrine records plus the two new starts.
- There are trailing END and stay-end bundles at 25.08 and 25.17.

The h slots have very large norms (about 874 each), while `out` has norm about 14 and z about 1.7. The slots probably hold large near-constant components, so dose information, if present, may be a small signal on top.

Next, the decoding test. For each glucose_finger reading at hour T ≥ 2, I take the state after the bundle just before it (normally the drug bundle at T − 1 + 5 min). Using cross-validated ridge regression, split by stay and standardising features, I predict:
- the last insulin rate, the insulin rate 2 and 3 hours back, and the last insulin residual relative to the sliding-scale mean;
- the last norepinephrine rate and its residual;
- the last glucose reading, as a positive control.

Separate fits on `out` alone and on the full state show where the information lives.

```python
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
sv = pd.DataFrame({'sid': st_sid, 'hour': st_hr, 'row': np.arange(len(st_sid))})
Pi = P.set_index(['stay_id','h'])
insmean_by_g = mean_by_g  # mean insulin rate per integer glucose (given insulin)
def ins_res(sid_, h_):
    u = Pi.insulin.get((sid_, h_), np.nan); g = Pi.g_now.get((sid_, h_), np.nan)
    if np.isnan(u): return 0.0
    return u - insmean_by_g.get(int(round(g)), np.nan)
rows=[]
for sid_, grp_ in sv.groupby('sid'):
    hrs = grp_.hour.values; rr = grp_.row.values
    for T in range(2, 25):
        if np.isnan(Pi.glucose_finger.get((sid_, T), np.nan)): continue
        prev = rr[hrs < T - 1e-6][-1]
        feat = {'row': prev, 'sid': sid_, 'T': T, 'g_last': Pi.g_now.get((sid_, T-1), np.nan)}
        for lag in [1,2,3]:
            feat[f'ins{lag}'] = np.nan_to_num(Pi.insulin.get((sid_, T-lag), np.nan))
            feat[f'ne{lag}'] = np.nan_to_num(Pi.norepinephrine.get((sid_, T-lag), np.nan))
        feat['insres1'] = ins_res(sid_, T-1)
        rows.append(feat)
D = pd.DataFrame(rows)
D['neres1'] = [Nres[np.where(sid==s_)[0][0], list(H).index(T-1)] for s_, T in zip(D.sid, D['T'])]
D = D.dropna()
Xs = X[D.row.values]; groups = D.sid.values
from sklearn.preprocessing import StandardScaler
def cv_r2(Xm, y):
    gkf = GroupKFold(5); pred = np.zeros_like(y)
    for tri, tei in gkf.split(Xm, y, groups):
        sc = StandardScaler().fit(Xm[tri]); m = RidgeCV(alphas=np.logspace(-2,4,13)).fit(sc.transform(Xm[tri]), y[tri])
        pred[tei] = m.predict(sc.transform(Xm[tei]))
    return 1 - ((y-pred)**2).mean()/y.var()
for tgt in ['g_last','ins1','ins2','ins3','insres1','ne1','ne2','neres1']:
    y = D[tgt].values.astype(float)
    print(tgt, "R2 out:", round(cv_r2(Xs[:, :192], y),3), " R2 all:", round(cv_r2(Xs, y),3))
```