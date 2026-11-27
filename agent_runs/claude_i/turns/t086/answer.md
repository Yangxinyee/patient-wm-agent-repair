The record has stay_id, hour, event, value, rate, dose and end_hour. Each insulin record starts at h+5 minutes and runs one hour. For every bundle row in fwd_3 I'll compute the insulin and norepinephrine rates started 0-1, 1-2, ..., 5-6 hours earlier. I'll also compute the kernel-weighted cumulative effect Σ rate × K(lag) for each drug. Then I'll cross-validate ridge regressions of these on the 992-number state, grouped by stay. A cross-validated R² above about 0.9 would mean the state carries the dose history linearly, so a head-side correction is feasible. A low R² would push me toward changing how doses enter the state.

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
X3s = z3['x'].astype(np.float64); sid3 = z3['stay_id']; hr3 = z3['hour']
tr3 = train[train.stay_id.isin(st3)]
def lagfeat(drug, K_):
    d = tr3[tr3.event==drug][['stay_id','hour','rate']].values
    out = np.zeros((len(sid3), 7))
    by = defaultdict(list)
    for s_, h_, r_ in d: by[int(s_)].append((h_, r_))
    for i in range(len(sid3)):
        for h_, r_ in by[int(sid3[i])]:
            lag = hr3[i] - h_
            if lag >= -1e-6 and lag < 6:
                out[i, int(lag + 1e-6)] += r_
            if lag >= -1e-6 and lag < 12:
                out[i, 6] += r_ * K_[min(int(lag + 1e-6), 11)]
    return out
Fi = lagfeat('insulin', K_ins_s); Fn = lagfeat('norepinephrine', K_ne_s)
Xs = (X3s - X3s.mean(0)) / (X3s.std(0) + 1e-6)
gkf = GroupKFold(5)
def cvr2(y, alpha=30.0, X=Xs):
    pr_ = np.zeros_like(y)
    for a_, b_ in gkf.split(X, y, sid3):
        mdl = Ridge(alpha=alpha).fit(X[a_], y[a_]); pr_[b_] = mdl.predict(X[b_])
    return 1 - ((y - pr_)**2).sum() / ((y - y.mean())**2).sum()
for nm_, Fx in [('ins', Fi), ('ne', Fn)]:
    print(nm_, [round(cvr2(Fx[:, j]), 3) for j in range(7)], 'sd', np.round(Fx.std(0), 2))
```