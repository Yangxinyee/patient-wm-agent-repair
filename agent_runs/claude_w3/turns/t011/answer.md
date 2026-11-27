The jitter looks like a randomized dose perturbation:
- Its lag-1 autocorrelation is 0.0007.
- The variance of stay means is 0.0063, against 0.0056 expected if the jitter were independent across hours, so any stay-level component is tiny.
- Its correlation with the glucose trend is 0.006.
- Its correlation with a simultaneous lab glucose, given the finger reading, is essentially zero.

There is a small correlation with position within band 1 (0.04), but that depends only on the recorded reading, which I can control for. So, given the recorded reading, the dose is as good as random. That lets me estimate the insulin effect from the records.

Next I will estimate the one-hour effect: regress the next finger reading on the current dose, with fixed effects for the current reading in 2 mg/dL bins, among hours with a fresh finger reading above 150. I will also show the raw mean change by dose within band 1. A clearly negative dose coefficient would mean insulin lowers glucose. A coefficient near zero, with tight confidence, would mean little or no one-hour effect. I will also note how often the next reading is present, since monitoring depends on glucose.

```python
pn = pn.sort_values(['stay_id','h'])
g = pn.groupby('stay_id')
pn['gf_next'] = g.glucose_finger.shift(-1)
pn['gl_next'] = g.glucose_lab.shift(-1)
pn['gf_prev'] = g.glucose_finger.shift(1)
base_mask = (pn.h>=0)&(pn.h<=23)&(pn.glucose_finger_age==0)&(pn.glucose_finger>150)
s = pn[base_mask].copy()
print("n:", len(s), " frac next finger present:", s.gf_next.notna().mean().round(4), " next finger or lab:", (s.gf_next.notna()|s.gl_next.notna()).mean().round(4))
s = s[s.gf_next.notna()].copy()
s['dG'] = s.gf_next - s.glucose_finger
# nonparametric within band 1
b1 = s[(s.glucose_finger>150)&(s.glucose_finger<=180)]
print(b1.groupby('insulin').agg(n=('dG','size'), dG=('dG','mean'), G=('glucose_finger','mean'), se=('dG', lambda x: x.std()/np.sqrt(len(x)))).round(2).to_string())
# FE regression: demean within 2-mg/dL bins of current reading
s['gbin'] = (s.glucose_finger//2).astype(int)
def fe_reg(df, y, xs, grp):
    d = df[[y]+xs+[grp]].dropna()
    dm = d[[y]+xs] - d.groupby(grp)[[y]+xs].transform('mean')
    X = dm[xs].values; Y = dm[y].values
    b = np.linalg.lstsq(X, Y, rcond=None)[0]
    res = Y - X@b
    # cluster-robust-ish (ignore clustering for now): simple se
    XtX_inv = np.linalg.inv(X.T@X)
    se = np.sqrt(np.diag(XtX_inv) * res.var())
    return b, se, len(d)
for band in [(150,180),(180,220),(220,260),(150,400)]:
    sb = s[(s.glucose_finger>band[0])&(s.glucose_finger<=band[1])]
    b, se, n = fe_reg(sb, 'dG', ['insulin'], 'gbin')
    print("band", band, " coef of dose on next-hour change:", b.round(3), " se", se.round(3), " n", n)
```