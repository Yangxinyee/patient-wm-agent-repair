`zpred` was defined with defaults GELU, `unit_row=2`, left-sided bins. That differs from the SiLU and right-sided values left in my globals. The earlier z-posterior match therefore used GELU, consistent with what the head search now shows.

Before chasing a "pending infusion" pathway, I should rule out a simpler explanation. NE patients have low MAP. My MAP error also correlates −0.78 with the last MAP reading. So the NE pattern may only be a weak persistence on the last value, for example a z-score or bin input for MAP that I compute differently.

Test: regress my MAP error on the last value's z and on the NE rate together.
- If the NE coefficient vanishes, the issue is the last-value pathway.
- If it survives, there is a real NE-dependent input I'm missing.

I also try rescaling the last-value z input, to see whether the persistence mismatch is just a scale error.

```python
q4['zl'] = fa4['zl'].values; q4['pm_mine'] = pm_
for ev in ['map_arterial','map_cuff','glucose_finger']:
    d_ = q4[q4.event==ev].copy(); d_['nor1'] = d_.nor_lag1.fillna(0); d_['ins1'] = d_.ins_lag1.fillna(0)
    Z = np.c_[np.ones(len(d_)), d_.zl, d_.nor1, d_.ins1]
    b_ = np.linalg.lstsq(Z, d_.err_pm.values, rcond=None)[0]; r_ = d_.err_pm.values - Z@b_
    se_ = np.sqrt(np.diag(np.linalg.inv(Z.T@Z))*r_.var())
    print(ev, 'err ~ 1+zl+nor1+ins1:', b_.round(2), '+-', se_.round(2), 'R2', round(1-r_.var()/d_.err_pm.var(),3))
for scl in [0.8, 1.2, 1.5]:
    f_ = f8_(); f_[:,2] *= scl
    pmS, _, nlS, _, _ = run12(f_)
    print('zl scale', scl, rep2(pmS, nlS), fa4.assign(ad=pmS-tgt).groupby('event').ad.mean().round(2).to_dict())
```