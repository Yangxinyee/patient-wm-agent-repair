Adjusting for history changes the estimated next-hour insulin effect a lot:
- naive slope −0.75 mg/dL per u/h, which is mostly regression to the mean through the sliding scale;
- with glucose bins and the previous rate, −0.39 (SE 0.04);
- adding the previous glucose change and hour, −0.015 (SE 0.04).

If the jitter depended only on the current reading, adding earlier history should not move the estimate this much. So either the jitter is related to past history (for example the trend or the previous rate, a titration rule), or the drug acts with a delay, or the jitter is a persistent per-stay multiplier.

Next I will characterise the jitter r = rate / base(G) where the scale applies. I will check
- whether it is persistent within stays (autocorrelation, stay-level variance share);
- whether it correlates with the glucose trend, the previous rate or the hour.

A persistent per-stay "aggressiveness" factor would be a natural instrument, provided it is unrelated to patient physiology. That last condition I can only partly test.

```python
P['base'] = np.where(P.gf.notna(), scale(P.gf.fillna(0)), np.where(P.gl.notna(), scale(P.gl.fillna(0)), 0))
P.loc[P.glu_any.isna(),'base'] = 0
P['basef'] = np.select([P.base==0],[np.nan], P.base.replace({0:np.nan}))
P['jit'] = P.ins / P.basef
P['lj'] = np.log(P.jit.where(P.jit>0))
g = P.groupby(level=0)
P['lj_prev'] = g.lj.shift(1)
dj = P[P.lj.notna()].copy()
print("jitter quantiles", dj.jit.quantile([.01,.05,.25,.5,.75,.95,.99]).round(3).to_dict())
print("corr lj with lj_prev:", dj[['lj','lj_prev']].corr().iloc[0,1])
# stay-level variance share
sm_ = dj.groupby(level=0).lj.agg(['mean','size','var'])
sm2 = sm_[sm_['size']>=5]
print("var of stay means", sm2['mean'].var(), "mean within var", sm2['var'].mean())
dj['dGprev'] = (dj.gf - g.gf.shift(1).reindex(dj.index))
dj['gfprev'] = g.gf.shift(1).reindex(dj.index)
for c in ['dGprev','ih','gf','ins_prev']:
    print(c, dj[['lj',c]].dropna().corr().iloc[0,1].round(4))
# jitter given base & dGprev
dj['tr'] = pd.cut(dj.dGprev, [-500,-30,-10,0,10,30,500])
print(dj.groupby(['base','tr']).jit.mean().unstack().round(3).to_string())
```