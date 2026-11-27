That run failed at the second line. The groupby object `gx` was built before I added the `map_last` column, so it could not see the column; nothing was computed. I will rebuild the groupby after adding columns and run the same predictability tests. The interpretation is the same as planned: any history variable with a large t-statistic would mean the innovations are partly predictable, which could produce the window-start bias.

```python
x['map_last'] = x.groupby('stay_id').map_pol.ffill()
gx = x.groupby('stay_id')
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
    for c, bb, ss in zip(['1']+cols, b, se): print(f"   {c:14s} {bb: .5f} (t={bb/ss: .1f})")
on_prev = x[(x.ne_prev>0)]
pred_test(on_prev[on_prev.fresh], ['map_pol','mp1','mp2','r2','r3','h','art_fresh'], "fresh, prev on")
pred_test(on_prev[~on_prev.fresh], ['map_last_prev','mp2','r2','h'], "no fresh, prev on")
q2s = q2.sort_values(['stay_id','h']).copy()
q2s['Ez'] = q2s.groupby(np.round(q2s.pol)).insulin.transform('mean'); q2s['z']= q2s.insulin - q2s.Ez
gq = q2s.groupby('stay_id')
q2s['pol_p1'] = gq.pol.shift(1); q2s['ins_p1'] = gq.insulin.shift(1); q2s['ins_p2']=gq.insulin.shift(2)
qq = q2s[q2s.pol>150][['z','pol','pol_p1','ins_p1','ins_p2','h']].dropna()
X = np.column_stack([np.ones(len(qq)), qq.pol, qq.pol_p1, qq.ins_p1, qq.ins_p2, qq.h])
b = np.linalg.lstsq(X, qq.z.values, rcond=None)[0]; res = qq.z.values - X@b; se=np.sqrt(np.diag(np.linalg.inv(X.T@X))*res.var())
print("insulin z on [1,pol,pol_p1,ins_p1,ins_p2,h]: t-stats", (b/se).round(1), " R2", round(1-res.var()/qq.z.var(),5))
```