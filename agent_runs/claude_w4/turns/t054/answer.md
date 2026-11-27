The context still does not match: mean error 0.04 against a cross-stay spread of 0.07, with large maximum errors under every variant. Too many hidden conventions remain (initial state, gap input, how the posterior is parameterised). Thirteen runs spent without convergence, so I am stopping the full CPU copy.

New plan: work with what I can measure.
- "states" with `both` shows how an edit changes the internal state at every bundle in factual mode.
- "factual" with `both` shows the change in predicted means and NLL.
- "dose_edit" with `both` gives the counterfactual readouts.

That supports roughly 15 measure-and-calibrate iterations.

The simplest candidate edit lives only in the heads. It reads recent insulin and norepinephrine information already in the state and shifts the value head's predicted mean. That works only if the state linearly encodes recent doses, and the dose surprises, well. So I will check with the fwd_3 states how well a cross-validated ridge readout of the state, at the bundle just before each reading hour T, recovers:
- recent insulin rates and surprises (lags 1–3);
- the latest glucose;
- the current norepinephrine rate.

High R² for rates and surprises keeps the heads-only route open; low R² means the core must be edited to write the dose in.

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
hr_r = np.round(hr_,4)
idx_read = [i for i in range(1,len(hr_r)) if sid_[i]==sid_[i-1] and float(hr_r[i]).is_integer() and 1 <= hr_r[i] <= 24]
prev = np.array(idx_read) - 1
Tser = T.set_index(['stay_id','h'])
def lagv(col, j):
    return np.array([Tser[col].get((sid_[i], int(hr_r[i]) - j), np.nan) for i in idx_read], float)
targets = {f'r{j}': np.nan_to_num(lagv('ins_rate', j)) for j in [1,2,3]}
targets.update({f's{j}': np.nan_to_num(lagv('s_ins', j)) for j in [1,2]})
targets['latG'] = lagv('latG', 1); targets['ne1'] = np.nan_to_num(lagv('ne_rate', 1))
grp = sid_[prev]
for feat_name, Xf in [('out', X_[prev,:192]), ('all', X_[prev,:960])]:
    for tn, yv in targets.items():
        okm = ~np.isnan(yv); Xs, ys, gs = Xf[okm], yv[okm], grp[okm]
        pred = np.zeros_like(ys)
        for trn, tst in GroupKFold(5).split(Xs, ys, gs):
            m_ = Ridge(alpha=10.0).fit(Xs[trn], ys[trn]); pred[tst] = m_.predict(Xs[tst])
        r2 = 1 - np.mean((ys-pred)**2)/np.var(ys)
        print(f'{feat_name:4s} {tn:5s} n={okm.sum():5d} sd={ys.std():7.3f}  CV R2={r2:.3f}')
```