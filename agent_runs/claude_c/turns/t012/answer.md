Insulin lowers next-hour glucose by about 1.0 mg/dL per U/h (−1.03 ± 0.04). The estimate is consistent across tiers (−1.05, −1.18, −0.76, −1.16), so the per-hour effect is small and roughly linear in rate. The tier dummies are negative even after controlling for the current dose. That hints that earlier doses still matter, i.e. effects persist and accumulate.

Next I will estimate the response at longer lags: the effect of the dose at hour s on the glucose at s+L, for L = 1..12. Conditional on the tier at s the dose is randomised, so the coefficient is a causal total effect. Mediation through later dosing should be negligible, because the step policy barely reacts to a 1 mg/dL shift. If the response stays at about −1 or grows with L, effects persist and accumulate over the 12-hour window. If it decays to 0 within a few hours, insulin's effect is transient.

```python
P['tf_or_lab'] = np.where(P.glucose_finger.notna(), tier(P.glucose_finger), np.where(P.glucose_lab.notna(), tier(P.glucose_lab), np.nan))
P['rd'] = P.glucose_finger.fillna(P.glucose_lab)
res = []
for L in range(1,13):
    P['GL'] = P.groupby(level=0).glucose_finger.shift(-L)
    q = P[P.rd.notna() & P.GL.notna()].copy()
    q = q[~q.rd.round().isin([150,180,220,260])]
    g = q.rd.values
    X = [np.ones(len(q)), g, g**2/100, g**3/1e4] + [(q.tf_or_lab.values==t).astype(float) for t in [1,2,3,4]] + [q.D.values]
    r = ols(q.GL.values, X, ['c','g','g2','g3','t1','t2','t3','t4','D'])
    res.append((L, len(q), r.loc['D','b'], r.loc['D','se']))
print(pd.DataFrame(res, columns=['L','n','b','se']).round(3))
```