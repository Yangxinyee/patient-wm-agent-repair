The insulin rule is now clear and gives a natural experiment.
- **Step:** the base is a step function of that hour's recorded glucose: finger if taken, else lab. It is 1, 2, 3, 4 or 6 u/h for 150–180, 180–220, 220–260, 260–300 and ≥300.
- **Multiplier:** the rate is base × m, rounded to 0.5, with m ≈ 1.04 ± 0.33 for every base.
- **Persistence:** m is mostly fresh each hour. Within-stay variance of log m is 0.12; between-stay variance of stay means is 0.010, against about 0.007 expected from pure noise, so any stay-level part is small.
- **Trigger:** the threshold is sharp at the recorded value, so the clinician acts on the recorded reading, not a hidden one.

So given the recorded glucose, the rate varies roughly at random. That lets me estimate the drug effect free of confounding by indication, provided m is unrelated to the patient's future course.

Next I will check the checkable assumptions:
- Does m (or the 3% omission) relate to observables such as MAP, previous rate or glucose trend?
- Does whether the next glucose is measured depend on the rate?

Then I will do the first effect estimate: next-hour finger glucose regressed on rate_t, holding the base and current glucose fixed.

```python
c = P.copy()
c['base_f'] = step_of(c.gf.fillna(-1)); 
c['Gdec'] = c.gf.where(c.gf.notna(), c.gl)   # glucose used for decision
c['base'] = step_of(c.Gdec.fillna(-1))
gg = c.groupby(level='stay_id')
for k in [1,2,3]:
    c[f'gf_p{k}'] = gg.gf.shift(-k); c[f'ins_p{k}']=gg.ins0.shift(-k)
c['gf_m1'] = gg.gf.shift(1); c['ma_m0']=c.ma_last
d = c[(c.gf.notna()) & (c.base>0) & c.ins.notna()].copy()
d['logm'] = np.log(d.ins/d.base)
# does multiplier relate to observables?
import statsmodels.api as sm_ if False else None
from sklearn.linear_model import LinearRegression
X = pd.DataFrame({'gf':d.gf, 'ins_prev':d.ins_prev, 'ma_last':d.ma_last.fillna(d.ma_last.mean()), 'nor0':d.nor0, 'hour':d.index.get_level_values('h'), 'gf_m1':d.gf_m1.fillna(-1), 'has_gfm1':d.gf_m1.notna().astype(float)})
for b in [1,2,3]:
    s = d.base==b
    lr = LinearRegression().fit(X[s], d.logm[s]); print('base',b,'R2', lr.score(X[s], d.logm[s]), dict(zip(X.columns, np.round(lr.coef_,5))))
# omission among triggered
t = c[(c.gf.notna())&(c.base_f>0)].copy()
t['omit']=t.ins.isna().astype(float)
print(t.groupby('base_f').omit.mean())
print(t.groupby(pd.cut(t.ma_last,[0,50,60,65,70,80,90,200]),observed=True).omit.agg(['mean','size']))
print(t.groupby(t.nor0>0).omit.mean())
# measurement of next gf vs rate within base
print(d.groupby(['base', pd.cut(d.ins/d.base,[0,0.6,0.9,1.1,1.4,5])],observed=True).gf_p1.apply(lambda x: x.notna().mean()).unstack().round(3))
```