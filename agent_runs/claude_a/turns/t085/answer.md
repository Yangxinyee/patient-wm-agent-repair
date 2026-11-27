The insulin effect looks linear and additive. The lag-1 slope per unit/h is almost the same across tiers: −1.94, −1.98, −2.03 and −2.17 at mean glucose 165, 196, 232 and 266. An effect proportional to glucose would have given about −3.1 in tier 4. The squared-rate terms are near zero. So my additive kernel (sum ≈ −11 per unit/h over about 8 h) is a sound target, and x½ should be close to −½ of x2.

Next I'll test norepinephrine's shape the same way. I'll regress next-hour arterial MAP on the current rate N within bins of the previous rate N_prev, controlling for current MAP, previous MAP and N_prev, and only where the infusion is continuing.
- A constant slope across N_prev bins means the effect is linear in rate: x2 ≈ −2 × x½.
- A slope that shrinks roughly as 1/N would mean a log-dose effect: x2 ≈ −x½.

```python
Hn = H[(H.N>0) & (H.N_prev>0) & H.map_now.notna() & H.ma_p1.notna() & H.map_prev.notna()].copy()
print("n", len(Hn), "N quantiles", np.round(Hn.N.quantile([.1,.25,.5,.75,.9]).values,3))
Hn['zone'] = np.where(Hn.map_now<65,0,np.where(Hn.map_now<75,1,2))
Hn['nb'] = pd.qcut(Hn.N_prev, 5, labels=False)
rows_out=[]
for nb_, g_ in Hn.groupby('nb'):
    for lag in [1,2]:
        yc = f'ma_p{lag}'; gg_ = g_[g_[yc].notna()]
        Z = pd.get_dummies(gg_.zone, prefix='z', drop_first=True).astype(float).values
        X_ = np.column_stack([np.ones(len(gg_)), gg_.N.values, gg_.map_now.values, gg_.map_prev.values, gg_.N_prev.values, Z])
        y_ = gg_[yc].values; b_, *_ = np.linalg.lstsq(X_, y_, rcond=None); r_ = y_-X_@b_
        se_ = np.sqrt(np.diag(np.linalg.inv(X_.T@X_))*r_.var())
        Xl = X_.copy(); Xl[:,1] = np.log(gg_.N.values); Xl[:,4] = np.log(gg_.N_prev.values)
        bl, *_ = np.linalg.lstsq(Xl, y_, rcond=None); rl = y_-Xl@bl; sel_ = np.sqrt(np.diag(np.linalg.inv(Xl.T@Xl))*rl.var())
        rows_out.append((nb_, lag, len(gg_), round(gg_.N_prev.mean(),3), round(gg_.N.mean(),3), round((np.log(gg_.N)-np.log(gg_.N_prev)).std(),3), round(b_[1],2), round(se_[1],2), round(bl[1],2), round(sel_[1],2), round(b_[1]*gg_.N.mean(),3)))
print(pd.DataFrame(rows_out, columns=['Nprev_bin','lag','n','Nprev','N','sd_logratio','slope_N','se','slope_logN','se_log','slope*N']).to_string())
```