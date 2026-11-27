The glucose placebo is reassuring: the dose at s does not predict earlier finger or lab glucose (coefficients 0.01–0.09, mostly within 2 SE). The lab outcome shows the same profile as the finger outcome: about −1 at 1–2 h, then positive, reaching +2.2 at 12 h. So the rebound is not finger-specific measurement noise.

The presence-of-reading test, however, fails the placebo. The dose at s "predicts" more earlier finger-sticks (+0.1 per U/h) and fewer later ones. The likely cause is lab-only hours. There the policy uses the lab tier with lower adherence (92% vs 97%), and those stays have fewer finger-sticks overall. So mixing lab-only hours with skips creates a non-exogenous source of dose variation.

Next I will separate the sources of variation:
1. Rate noise among given doses only (D > 0), in hours where the finger-stick set the tier.
2. Skip versus given, in the same hours.

If the rebound persists using only rate noise among given doses, it is a real causal feature. If it comes from skips or lab-only hours, it is probably confounding.

```python
res = []
for L in list(range(-4,0)) + [1,2,3,4,6,8,10,12]:
    P['GL'] = P.groupby(level=0).glucose_finger.shift(-L)
    q = P[P.glucose_finger.notna() & (P.tf>=1)].copy()
    q = q[~q.glucose_finger.round().isin([150,180,220,260])]
    q['given'] = q.insulin.notna().astype(float)
    g = q.glucose_finger.values
    base = [np.ones(len(q)), g, g**2/100, g**3/1e4] + [(q.tf.values==t).astype(float) for t in [2,3,4]]
    nm = ['c','g','g2','g3','t2','t3','t4']
    pres = q.GL.notna().astype(float).values
    # among given: rate noise
    m = (q.given.values==1)
    mm = m & q.GL.notna().values
    r1 = ols(q.GL.values[mm], [c[mm] for c in base]+[q.D.values[mm]], nm+['D']).loc['D']
    p1 = ols(pres[m], [c[m] for c in base]+[q.D.values[m]], nm+['D']).loc['D']
    # skip vs given (given indicator x base)
    mm2 = q.GL.notna().values
    bt = q.tf.map({1:1,2:2,3:3,4:4.4}).values
    r2 = ols(q.GL.values[mm2], [c[mm2] for c in base]+[(q.given.values*bt)[mm2]], nm+['Gv']).loc['Gv']
    p2 = ols(pres, base+[q.given.values*bt], nm+['Gv']).loc['Gv']
    res.append((L, r1.b, r1.se, p1.b, p1.se, r2.b, r2.se, p2.b, p2.se))
print(pd.DataFrame(res, columns=['L','rate_b','se','rate_pres','se','skip_b(per U/h)','se','skip_pres','se']).round(4).to_string())
```