The insulin policy is a sliding scale on the latest glucose_finger:
- essentially never below 140;
- 37% of the time between 140 and 160;
- always above 160, at about 1 U/h for 160-180, 2 for 180-220, 3 for 220-250, 3.7 for 250-300, and 6.5 above 300.

The rate also varies a lot from hour to hour at the same glucose, which the transition table shows is not pure persistence. NE is graded on MAP: always given below 60, at about 60% for 65-70, rare above 80, with higher rates at lower MAP. Strong treatment-by-indication confounding is clearly present.

Next I'll pin down how deterministic the policy is. I'll look at P(insulin) and the rate distribution in fine glucose bins (1 mg/dL around 140-170), and check whether the reading used is the finger value at the same hour or a lab value. A sharp threshold would allow a regression-discontinuity design. A fuzzy one suggests the clinician sees something I don't (hidden confounding), or that there is deliberate randomness.

```python
y = x[x.gf.notna()].copy()  # hours with a fresh finger reading
y['gfi'] = y.gf.round()
tab = y[(y.gfi>=140)&(y.gfi<=200)].groupby('gfi').agg(p=('ins', lambda s: s.notna().mean()), rate=('ins','mean'), sd=('ins','std'), n=('ins','size'))
print(tab.to_string())
# hours without a fresh gf: what does the policy follow?
z = x[x.gf.isna() & x.gf_ff.notna()].copy()
print('no fresh gf: n', len(z), 'p_ins by gf_ff band'); z['b']=pd.cut(z.gf_ff,[0,140,150,155,160,165,170,180,400])
print(z.groupby('b',observed=True).agg(p=('ins', lambda s: s.notna().mean()), n=('ins','size')).to_string())
```