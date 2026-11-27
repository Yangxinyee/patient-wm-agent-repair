Splitting the variation resolves the puzzle: the "rebound" was an artefact of skipped doses.
- **Rate noise among given doses** passes the placebo tests: no relation to earlier finger-stick presence, and only about +0.08 ± 0.05 on earlier glucose. Its effect is about −2.0 mg/dL per U/h at 1 h, −2.2 at 2 h, −1.75 at 3 h, −1.27 at 4 h, −0.45 at 6 h and −0.14 at 8 h, reaching about 0 by 10–12 h. Insulin lowers glucose, peaks at about 2 h and wears off over about 6–8 h.
- **Skips** are heavily confounded. Given-versus-skipped "predicts" earlier glucose (+10 mg/dL) and earlier finger-stick presence (+0.45). Skips cluster in hours with little prior monitoring, so they are not a valid experiment.

If this response holds and is linear, a sustained doubling from hour 12 lowers glucose by a cumulative sum of about −9 mg/dL per U/h of dose. That makes the true ×2 readout something like −10 to −15 mg/dL.

Before relying on linear extrapolation to doubled doses, I need to check:
1. Is the effect linear in dose, or does it saturate? I will compare dose categories within a tier.
2. Does it depend on the glucose level? I will estimate it by tier.
3. Where do skips occur? This matters for interpreting the data.

```python
P['GL1'] = P.groupby(level=0).glucose_finger.shift(-1)
P['GL2'] = P.groupby(level=0).glucose_finger.shift(-2)
q = P[P.glucose_finger.notna() & (P.tf>=1) & P.insulin.notna()].copy()
q = q[~q.glucose_finger.round().isin([150,180,220,260])]
for t in [1,2,3,4]:
    for out in ['GL1','GL2']:
        qt = q[(q.tf==t) & q[out].notna()]
        gt = qt.glucose_finger.values
        r = ols(qt[out].values, [np.ones(len(qt)), gt, qt.D.values], ['c','g','D']).loc['D']
        print('tier',t,out,len(qt), r.values.round(3))
# dose categories within tier 2 and 3 (residualised on g)
for t in [2,3]:
    qt = q[(q.tf==t) & q.GL1.notna() & q.GL2.notna()].copy()
    gt = qt.glucose_finger.values
    X = np.column_stack([np.ones(len(qt)), gt]); 
    for out in ['GL1','GL2']:
        b,*_ = lstsq(X, qt[out].values, rcond=None)
        qt['res_'+out] = qt[out].values - X@b
    print(qt.groupby(qt.D.clip(upper=7)).agg(n=('D','size'), r1=('res_GL1','mean'), r2=('res_GL2','mean')).round(2).T.to_string())
# skips
sk = P[P.glucose_finger.notna() & (P.tf>=1) & P.insulin.isna() & ~P.glucose_finger.round().isin([150,180,220,260])]
print("skip hours by h:", sk.index.get_level_values('h').value_counts().sort_index().to_dict())
print("prev insulin present among skips:", sk.ins_prev.notna().mean(), " overall given:", P[P.glucose_finger.notna() & (P.tf>=1) & P.insulin.notna()].ins_prev.notna().mean())
```