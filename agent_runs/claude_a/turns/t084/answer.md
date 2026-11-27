`H` is my stay-hour panel for the 30,000 training stays: glucose and MAP at hour h, insulin rate `I`, norepinephrine rate `N`, tier, leads of glucose (`gf_p*`), MAP (`ma_p*`) and doses, and the lagged glucose `gf_m1`. Finger glucose is missing at some hours.

Next I'll test the shape of the insulin effect. Within each glucose tier, the rate varies only through the random multiplier. So I'll regress glucose one hour later on the rate, controlling for current and previous glucose, separately per tier. A clean causal slope at lag 1 needs no feedback correction. I'll also add a squared-rate term within tier to test curvature.

- A roughly constant slope (≈ −2 per unit/h) across tiers would support a linear, additive effect.
- A slope growing with the tier's glucose level (~165 → ≥250) would mean the effect scales with glucose.
- A significant square term would mean nonlinearity in the dose.

```python
Hq = H[(H.tier>=1) & H.glucose_finger.notna() & H.gf_p1.notna() & H.gf_m1.notna()].copy()
print("n rows", len(Hq))
rows_out = []
for tq in [1,2,3,4]:
    g_ = Hq[Hq.tier==tq]
    for lag in [1,2,3]:
        ycol = f'gf_p{lag}'; gg_ = g_[g_[ycol].notna()]
        X_ = np.column_stack([np.ones(len(gg_)), gg_.I.values, gg_.glucose_finger.values, gg_.gf_m1.values])
        y_ = gg_[ycol].values
        b_, *_ = np.linalg.lstsq(X_, y_, rcond=None); res_ = y_ - X_@b_
        se_ = np.sqrt(np.diag(np.linalg.inv(X_.T@X_)) * res_.var())
        # curvature test: add (I - mean)^2
        Ic = gg_.I.values - gg_.I.values.mean()
        X2 = np.column_stack([X_, Ic**2]); b2_, *_ = np.linalg.lstsq(X2, y_, rcond=None); r2_ = y_-X2@b2_
        se2 = np.sqrt(np.diag(np.linalg.inv(X2.T@X2))*r2_.var())
        rows_out.append((tq, lag, len(gg_), round(gg_.glucose_finger.mean(),1), round(gg_.I.mean(),2), round(gg_.I.std(),2), round(b_[1],3), round(se_[1],3), round(b2_[4],3), round(se2[4],3)))
print(pd.DataFrame(rows_out, columns=['tier','lag','n','G_mean','I_mean','I_sd','slope_I','se','quad','se_quad']).to_string())
```