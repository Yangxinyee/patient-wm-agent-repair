The one-hour insulin effect is very stable, close to −2.0 mg/dL per u/h, whether or not norepinephrine is running and early, mid or late in the stay. Cross-effects are negligible: insulin moves next-hour MAP by +0.01 mmHg per u/h, and norepinephrine moves glucose by +0.47 mg/dL per mcg/kg/min, which is tiny at real rates.

Without insulin, glucose drifts up slowly at normal levels (+1 to +3 mg/dL per hour). The puzzle is the hours where insulin was triggered but omitted. There glucose then falls, by about 3.4 mg/dL at 180–200 and 7.4 at 200–250. That is more than it falls with insulin given, so omission is informative: clinicians seem to skip insulin when something unrecorded says glucose will fall. This does not affect the rate-noise estimate, but I want to understand it: it could be a hidden confounder or a sign of spurious readings.

Next I will compare the distribution of next-hour changes after omitted and given hours at similar glucose (a mixture with big drops would point to outlier readings). I will also look at what precedes omissions (earlier readings, the lab-versus-finger gap).

```python
t = c[(c.gf.notna())&(c.base_f>0)&c.gf_p1.notna()].copy(); t['omit']=t.ins.isna()
t['d1']=t.gf_p1-t.gf
for b in [1,2,3]:
    s=t[t.base_f==b]
    print('base',b, 'omit n',s.omit.sum(), 'quantiles d1 omitted', np.round(np.quantile(s.d1[s.omit],[.05,.25,.5,.75,.95]),1), ' given', np.round(np.quantile(s.d1[~s.omit],[.05,.25,.5,.75,.95]),1))
# is the reading at t an outlier relative to previous reading? compare gf - gf_m1
t['jump'] = t.gf - t.gf_m1
print(t.groupby('omit').jump.describe())
# lab at same hour vs finger for omitted
tl = t[t.gl.notna()]
print('finger-lab diff by omit:', tl.groupby('omit').apply(lambda x: (x.gf-x.gl).describe()).round(2).to_string())
```