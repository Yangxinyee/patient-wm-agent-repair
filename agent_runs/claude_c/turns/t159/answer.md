The zero-token model shifts glucose predictions by only 1.1 mg/dL on average (mean −0.63) and MAP by 0.3 mmHg. That is tiny next to the true drug effects of several mg/dL per unit-hour. To put the token's potency in per-unit terms, I'll regress the prediction shift on the rates given 1–6 hours before each reading. The original model's per-unit effect at each lag is minus each coefficient. If the lag coefficients come out near +0.2–0.3 for insulin, about 1/7 of the true −2, then the token's entire potency is weak, not just its rate dependence.

```python
tu['h0'] = np.floor(tu.hour).astype(int)
def ratemat(df, drug):
    r = df[df.event==drug].groupby(['stay_id','h0']).rate.sum()
    return r
ri = ratemat(tu,'insulin'); rn = ratemat(tu,'norepinephrine')
def lagX(sub, r, L=6):
    X = np.zeros((len(sub), L))
    for l in range(1, L+1):
        idx = list(zip(sub.stay_id, sub.hour - l))
        X[:, l-1] = r.reindex(idx).fillna(0).values
    return X
for ev, r in [('glucose_finger', ri), ('map_arterial', rn)]:
    sub = m[m.event==ev]
    X = lagX(sub, r); X1 = np.c_[np.ones(len(X)), X]
    b = np.linalg.lstsq(X1, sub.d.values, rcond=None)[0]
    print(ev, np.round(b, 3))
```