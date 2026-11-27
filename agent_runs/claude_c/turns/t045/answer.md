The bundle structure is now clear. Readings sit at integer hours and drug bundles at h+0.0833; in the 100 tuning stays there are 1,024 insulin bundles with mean rate 1.66 U/h. The finger glucose at the same hour is available for 97% of them, so I can compute the policy's expected rate.

Next I build the insulin dose innovation, the recorded rate minus the mean rate of its glucose tier (tier means from the training panel). Then I regress the 32-dimensional z posterior mean at insulin bundles on the innovation and on the expected rate.

- If z has a direction that tracks the innovation strongly, with little loading on the expected rate, scaling `z_in` along it is a candidate edit that avoids double counting.
- If z mostly tracks the raw or expected rate, that route fails and I need another one.

```python
def base_of(g):
    g = np.asarray(g, float)
    b = np.zeros_like(g)
    b[g >= 150] = 1; b[g >= 180] = 2; b[g >= 220] = 3; b[g >= 260] = 4
    return b
# training tier means of given insulin (finger-tier hours)
Pt_ = P[['glucose_finger','glucose_lab','insulin']].copy()
gsrc = Pt_['glucose_finger'].where(Pt_['glucose_finger'].notna(), Pt_['glucose_lab'])
Pt_['tierk'] = base_of(gsrc.values)
Pt_ = Pt_[gsrc.notna()]
tm = Pt_.groupby('tierk')['insulin'].agg(['mean','std','count', lambda x: (x>0).mean()])
print(tm)
tier_mean = tm['mean'].to_dict()
gsrc_b = np.where(np.isnan(gf), gl, gf)
tk = base_of(gsrc_b)
exp_rate = np.array([tier_mean.get(t, np.nan) for t in tk])
innov = rate - exp_rate
Zi = X3[ins_b, 960:992]
ok = ~np.isnan(innov) & ~np.isnan(gsrc_b)
print("n ok", ok.sum(), "innov sd", np.nanstd(innov[ok]), "exp_rate sd", np.nanstd(exp_rate[ok]))
# regress each z on [1, innov, exp_rate, g]
A = np.column_stack([np.ones(ok.sum()), innov[ok], exp_rate[ok], (gsrc_b[ok]-200)/50])
coef, *_ = np.linalg.lstsq(A, Zi[ok], rcond=None)
res_ = Zi[ok] - A @ coef
r2 = 1 - res_.var(0) / Zi[ok].var(0)
print("z var at ins bundles (top):", np.round(np.sort(Zi[ok].var(0))[::-1][:6],3))
# variance explained by each regressor overall
for j, nmj in enumerate(['innov','exp_rate','g']):
    A2 = np.delete(A, j+1, axis=1)
    c2, *_ = np.linalg.lstsq(A2, Zi[ok], rcond=None)
    r2b = 1 - (Zi[ok]-A2@c2).var(0).sum()/Zi[ok].var(0).sum()
    print(nmj, "drop-one total-R2 loss:", round((1-res_.var(0).sum()/Zi[ok].var(0).sum()) - r2b, 4))
# best linear predictor of innovation from z
Az = np.column_stack([np.ones(ok.sum()), Zi[ok]])
cz, *_ = np.linalg.lstsq(Az, innov[ok], rcond=None)
pred_in = Az @ cz
print("R2 innov ~ z:", 1 - np.var(innov[ok]-pred_in)/np.var(innov[ok]))
cz2, *_ = np.linalg.lstsq(Az, exp_rate[ok], rcond=None)
print("R2 exp_rate ~ z:", 1 - np.var(exp_rate[ok]-Az@cz2)/np.var(exp_rate[ok]))
cz3, *_ = np.linalg.lstsq(Az, rate[ok], rcond=None)
print("R2 rate ~ z:", 1 - np.var(rate[ok]-Az@cz3)/np.var(rate[ok]))
```