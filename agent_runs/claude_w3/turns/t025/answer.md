Norepinephrine raises arterial MAP in these records, transiently. Per 1 mcg/kg/min:

| lag (h) | effect (mmHg) |
|---|---|
| 1 | 2.7–3.4 |
| 2 | 2.5–2.9 |
| 3 | about 1.6–1.9 |
| 4 | about 0.9 |
| 5 | about 0.5 |
| 6 and later | about 0 |

Summed, that is a gain of about 9 mmHg per mcg/kg/min.

One pattern makes me distrust the long lags. In every window of 6 hours or more, the coefficient on the first dose comes out negative (−1 to −1.5), while the same lag is about 0 when it is not first in the window. Insulin showed the mirror pattern: the first coefficient was weaker than the same lag in longer windows. A genuine rebound would not depend on window position, so this is more likely a bias in my design. A likely cause is that my innovation is partly predictable from history the policy uses but my cells omit, such as the last MAP in hours without a fresh reading, or older MAP and rates.

Next I will regress the norepinephrine innovation zr on lagged MAP readings, lagged rates, the hour and reading type, separately for fresh and no-fresh hours. I will do the same for the insulin jitter. If something predicts the innovation, I will add it to the cells and re-estimate. If nothing does, the anomaly must come from elsewhere, such as nonlinearity.

```python
x['map_last'] = gx.map_pol.ffill()
x['map_last_prev'] = gx.map_last.shift(1)
for lag in [1,2,3]:
    x[f'mp{lag}'] = gx.map_pol.shift(lag)
    x[f'r{lag}'] = gx.nr.shift(lag)
x['art_fresh'] = x.map_arterial.notna().astype(int)
def pred_test(df, cols, nm):
    d = df[['zr']+cols].dropna()
    X = np.column_stack([np.ones(len(d))]+[d[c].values for c in cols])
    b = np.linalg.lstsq(X, d.zr.values, rcond=None)[0]
    res = d.zr.values - X@b
    se = np.sqrt(np.diag(np.linalg.inv(X.T@X))*res.var())
    print(nm, "n", len(d), " R2", round(1-res.var()/d.zr.var(),4))
    for c, bb, ss in zip(['1']+cols, b, se): print(f"   {c:12s} {bb: .5f} (t={bb/ss: .1f})")
on_prev = x[(x.ne_prev>0)]
pred_test(on_prev[on_prev.fresh], ['map_pol','mp1','mp2','r2','r3','h','art_fresh'], "fresh, prev on")
pred_test(on_prev[~on_prev.fresh], ['map_last_prev','mp2','r2','h'], "no fresh, prev on")
# insulin jitter predictability
q2s = q2.sort_values(['stay_id','h']).copy()
gq = q2s.groupby('stay_id')
q2s['Ez'] = q2s.groupby(np.round(q2s.pol)).insulin.transform('mean'); q2s['z']= q2s.insulin - q2s.Ez
q2s['pol_p1'] = gq.pol.shift(1); q2s['ins_p1'] = gq.insulin.shift(1); q2s['ins_p2']=gq.insulin.shift(2)
qq = q2s[q2s.pol>150][['z','pol','pol_p1','ins_p1','ins_p2','h']].dropna()
X = np.column_stack([np.ones(len(qq)), qq.pol, qq.pol_p1, qq.ins_p1, qq.ins_p2, qq.h])
b = np.linalg.lstsq(X, qq.z.values, rcond=None)[0]; res = qq.z.values - X@b; se=np.sqrt(np.diag(np.linalg.inv(X.T@X))*res.var())
print("insulin z on [1,pol,pol_p1,ins_p1,ins_p2,h]: t-stats", (b/se).round(1), " R2", round(1-res.var()/qq.z.var(),5))
```