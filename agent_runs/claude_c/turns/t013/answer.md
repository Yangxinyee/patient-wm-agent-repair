The longer-lag profile is surprising. A dose lowers glucose by about 1.0 mg/dL at 1–2 h (−0.35 at 3 h). The sign then reverses: +0.2 at 4 h, rising steadily to about +2 mg/dL per U/h at 12 h. Taken at face value, a sustained insulin increase lowers glucose for a few hours and then raises it. Policy mediation cannot explain this; E[dose | G] changes only about 0.02 U/h per mg/dL.

Three explanations remain:
1. A real delayed rebound built into this simulator.
2. Hidden confounding: the dose noise correlates with something that raises later glucose.
3. Selection bias: whether a finger-stick is taken at s+L may depend on dose or glucose.

Next I will run checks to discriminate:
- **Placebo:** does the dose at s "predict" glucose before s, given the reading at s? A null placebo supports exogeneity.
- **Measurement process:** does the chance of a finger-stick depend on the current dose or reading?
- **Lab glucose:** does it, as an outcome, show the same rebound?

```python
res = []
for L in list(range(-6,0)) + [1,2,4,6,8,10,12]:
    P['GL'] = P.groupby(level=0).glucose_finger.shift(-L)
    P['LL'] = P.groupby(level=0).glucose_lab.shift(-L)
    q = P[P.rd.notna()].copy()
    q = q[~q.rd.round().isin([150,180,220,260])]
    g = q.rd.values
    base = [np.ones(len(q)), g, g**2/100, g**3/1e4] + [(q.tf_or_lab.values==t).astype(float) for t in [1,2,3,4]] + [q.D.values]
    nm = ['c','g','g2','g3','t1','t2','t3','t4','D']
    m = q.GL.notna().values
    rf = ols(q.GL.values[m], [c[m] for c in base], nm).loc['D']
    m2 = q.LL.notna().values
    rl = ols(q.LL.values[m2], [c[m2] for c in base], nm).loc['D']
    # presence of finger reading at s+L
    pres = q.GL.notna().astype(float).values
    rp = ols(pres, base, nm).loc['D']
    res.append((L, rf.b, rf.se, rl.b, rl.se, rp.b, rp.se))
print(pd.DataFrame(res, columns=['L','finger_b','se','lab_b','se_l','presence_b','se_p']).round(4))
```