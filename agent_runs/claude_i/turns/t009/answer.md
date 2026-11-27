This confirms the insulin policy is driven by the recorded finger glucose:
- When finger and lab readings coexist in an hour, the finger reading agrees with the dosing decision 99.6% of the time.
- A recorded finger value of 150 gives insulin 50% of the time, so the threshold sits at 150 on the *unrounded* reading.
- Insulin is never given in an hour without a reading.
- Within each band the rate scatters widely, which looks like multiplicative noise rounded to 0.5. In the 150–180 band, for example, rates are 0.5, 1.0, 1.5 or 2.0, with 1.0 the mode.

If this scatter is random, it is a natural experiment. Patients with the same reading get different doses, so comparing their next readings measures what insulin does. The 150 threshold gives a second, independent design (a regression discontinuity).

Next I'll check three things:
1. Whether the within-band rate deviation is iid or persists within a stay. Persistence would hint at a hidden patient factor.
2. How finger and lab readings relate (their measurement noise and bias).
3. The first rough estimate of the next-hour effect of rate, controlling for the current reading.

```python
Q = P[P.index.get_level_values('h')>=0].copy()
# gl-only hours: does gl drive insulin?
glo = Q[Q.gf.isna() & Q.gl.notna()]
print("gl-only hours:", len(glo), " p(ins)", glo['ins'].notna().mean().round(3), " agree gl>=150:", ((glo.gl>=150)==glo['ins'].notna()).mean().round(4))
# measurement relation gf vs gl
b = Q[Q.gf.notna() & Q.gl.notna()]
d = b.gf-b.gl
print("gf-gl: mean %.2f sd %.2f; corr %.3f"%(d.mean(), d.std(), np.corrcoef(b.gf,b.gl)[0,1]))
print("gf-gl by gl level:", d.groupby(pd.cut(b.gl,[0,100,130,160,200,250,500]),observed=True).agg(['mean','std','size']).round(2).to_dict())
# band base and log deviation
def base(g):
    return np.select([g<150, g<180, g<220, g<250], [0,1.0,2.0,3.0], default=np.nan)
Q['gdec'] = Q.gf
Q['ins_band'] = pd.cut(Q.gdec, [150,180,220,250,1000], right=False)
qi = Q[Q['ins'].notna() & Q.gdec.notna()].copy()
qi['ldev'] = np.log(qi['ins']) - qi.groupby('ins_band', observed=True)['ins'].transform(lambda x: np.log(x).mean())
# within-stay autocorrelation of deviation: consecutive insulin hours
qi = qi.reset_index()
qi['ldev_prev'] = qi.groupby('stay_id').ldev.shift(1)
qi['h_prev'] = qi.groupby('stay_id').h.shift(1)
cc = qi.dropna(subset=['ldev_prev'])
print("autocorr of rate log-deviation (consecutive dosing hours):", np.corrcoef(cc.ldev, cc.ldev_prev)[0,1].round(4), len(cc))
# stay-level mean deviation variance vs expected under iid
sm = qi.groupby('stay_id').ldev.agg(['mean','size'])
sm = sm[sm['size']>=5]
print("var of stay-mean dev:", sm['mean'].var().round(4), " expected iid:", (qi.ldev.var()/sm['size']).mean().round(4))
```