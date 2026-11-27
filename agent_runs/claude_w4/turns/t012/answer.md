## Step 11: test the within-band noise as a natural experiment

**What the last result showed.** The sliding scale uses the most recent glucose reading, with finger winning when finger and lab fall in the same hour. That rule matches insulin on/off status in 99.7% of hours. The policy is now nearly fully specified:

rate = 0 if latest glucose ≤ 150, else base(band) × i.i.d. multiplicative noise, rounded to 0.5.

**What I will do next.** Estimate the one-hour effect of insulin.
- Use hours with a fresh finger reading at t and another finger reading at t+1.
- Regress G(t+1) on the insulin rate at t.
- Control flexibly for the reading at t (spline and band dummies), the previous readings and doses, MAP and norepinephrine.

Because the rate's within-band variation is i.i.d. noise, this coefficient should be causal if the noise does not depend on hidden state. I will also run placebo checks:
- the current noise against the past reading change G(t) − G(t−1);
- the current noise against the previous hour's dose.

**What would count as evidence.** A clean negative coefficient with null placebos would mean insulin lowers glucose, roughly linearly per unit/h. A coefficient near zero would mean little or no effect. A significant placebo would warn of confounded noise.

```python
x = g[g.h.between(0,24)].copy()
x['lg'] = latest(x,'finger')
x['band'] = band_of(x.lg.values)
x['gf_next'] = x.groupby('stay_id').glucose_finger.shift(-1)
x['gl_next'] = x.groupby('stay_id').glucose_lab.shift(-1)
x['gf_prev'] = x.groupby('stay_id').glucose_finger.shift(1)
x['ins_prev2'] = x.groupby('stay_id').ins_rate.shift(2).fillna(0)
base_rate = np.array([0,1,2,3,4.0])
x['noise'] = np.where(x.ins_rate>0, np.log(np.maximum(x.ins_rate,0.5)/np.maximum(base_rate[x.band],1e-9)), np.nan)
s = x[(x.age_glucose_finger==0) & (x.ins_rate>0) & x.gf_next.notna()].copy()
import numpy.linalg as la
def ols(y, X, names):
    X = np.column_stack([np.ones(len(y))]+X)
    b, *_ = la.lstsq(X, y, rcond=None)
    r = y - X@b
    s2 = r@r/(len(y)-X.shape[1])
    cov = s2*la.inv(X.T@X)
    se = np.sqrt(np.diag(cov))
    return pd.DataFrame({'b':b,'se':se}, index=['const']+names)
gf = s.glucose_finger.values
spl = [np.maximum(gf-k,0) for k in [120,150,165,180,200,220,240,260,280]]
bd = [(s.band.values==k).astype(float) for k in [2,3,4]]
res = ols(s.gf_next.values, [s.ins_rate.values, gf]+spl+bd+[s.prev_ins_rate.values, s.ins_prev2.values], ['ins','gf']+[f'sp{k}' for k in range(9)]+['b2','b3','b4','ins_prev','ins_prev2'])
print(res.round(3).to_string())
print("n=",len(s))
# placebo: noise vs past change and prev dose
p = s[s.gf_prev.notna()]
print("corr(noise, gf - gf_prev):", np.corrcoef(p.noise, p.glucose_finger-p.gf_prev)[0,1], " corr(noise, prev_ins_rate):", np.corrcoef(s.noise, s.prev_ins_rate)[0,1])
print("corr(noise, last_map_cuff):", s[['noise','last_map_cuff']].dropna().corr().iloc[0,1], " corr(noise, ne_rate):", np.corrcoef(s.noise, s.ne_rate)[0,1])
```