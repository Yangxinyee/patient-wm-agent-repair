Two findings:
1. In hours with a fresh MAP, the norepinephrine innovation is unpredictable (R² ≈ 0, no t above 2). In hours without a fresh MAP it is predictable from the last MAP (t = −15.7, R² 0.034). There the policy titrates on the most recent MAP, so my cells for those hours were wrong. That makes those instruments invalid and plausibly explains the negative window-start coefficient.
2. The insulin jitter is unpredictable from everything I tried (R² 0.00001). The milder insulin window pattern is therefore probably noise, or effects that change over time or with state, not instrument failure.

Next I will rebuild the norepinephrine cells. Fresh hours keep (MAP bin, previous-rate bin). No-fresh hours use (last-MAP bin, age of last MAP capped at 3, previous-rate bin). I will recheck that the new innovation is unpredictable in no-fresh hours, then rerun the norepinephrine IV. If the negative first-lag coefficients disappear, the fix worked and the corrected impulse response is my norepinephrine estimate.

```python
x['map_age'] = x['h'] - x['h'].where(x.map_pol.notna()).groupby(x.stay_id).ffill()
x['mapb2'] = np.where(x.fresh, np.clip(np.round(x.map_pol/2)*2, 30, 110),
               np.where(x.map_last.notna(), 1000 + np.clip(np.round(x.map_last/2)*2, 30, 110)*10 + np.minimum(x.map_age.fillna(9),3), -1))
key2 = x.mapb2.astype(str)+'_'+x.prevb.astype(str)
x['Er2'] = x.groupby(key2).nr.transform('mean'); x['zr2'] = x.nr - x.Er2
x['zr'] = x['zr2']
on_prev = x[(x.ne_prev>0)]
pred_test(on_prev[~on_prev.fresh], ['map_last_prev','mp2','r2','h'], "no fresh, prev on (new cells)")
pred_test(on_prev[on_prev.fresh], ['map_pol','mp1','mp2','r2','r3','h','art_fresh'], "fresh, prev on (new cells)")
ZR = x['zr2'].values.reshape(SS,25)
KEYC = pd.factorize(key2)[0].reshape(SS,25)
for L in [1,2,3,4,6,8,10,12]:
    b, se, n = iv_ne(L)
    print(f"L={L:2d} n={n:6d} coef r_t (lag L): {b[0]:7.2f} ({se[0]:.2f}); by lag (L..1):", np.round(b,1))
```