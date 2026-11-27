Baseline glucose behaves like a slowly mean-reverting latent process:
- a stay-level random effect;
- an AR component with hourly persistence of roughly 0.93–0.95;
- measurement noise with SD around 5.

Off insulin, finger-sticks occur at random about 30% of the time with no autocorrelation, and lab draws are iid at 20%. So all measurement and dosing decisions depend only on recorded information. That makes a likelihood-based state-space fit valid.

The dose-response shape is still unclear. Within band 1 the next-hour effect accelerates with rate (0.5→1→1.5→2: −0.73, −1.81, −3.07 relative to 0.5). Within band 2 it is roughly linear at about −2 per u/h above 1.5, but flat from 1.0 to 1.5.

Before building the full kernel model I'll sharpen this. I'll use rate dummies within each band, add controls for the previous glucose and previous rate to cut noise, and look at lags 1–3. That shows whether the low-rate deficit is real, or a timing effect that catches up later.

```python
Q = P.copy()
for k in range(1,4):
    Q[f'gf_lead{k}'] = Q.groupby(level='stay_id').gf.shift(-k)
    Q[f'gl_lead{k}'] = Q.groupby(level='stay_id').gl.shift(-k)
    Q[f'g_lead{k}'] = Q[f'gf_lead{k}'].combine_first(Q[f'gl_lead{k}'])
Q['gdec'] = Q.gf.combine_first(Q.gl)
Q['g_prev'] = Q.groupby(level='stay_id').g_last.shift(1)
Q = Q[Q.index.get_level_values('h')>=1]
bands = {1:(150,180,[0.5,1.0,1.5,2.0,2.5]), 2:(180,220,[1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5]), 3:(220,250,[1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0])}
for bnum,(lo,hi,rates) in bands.items():
    s = Q[(Q.gdec>=lo)&(Q.gdec<hi)&Q['ins'].isin(rates)&Q.g_prev.notna()]
    for k in [1,2,3]:
        X = [s.gdec, s.g_prev, s.ins_prev] + [(s['ins']==r).astype(float) for r in rates[1:]]
        res, n, sd = ols(s[f'g_lead{k}'], X, ['gdec','gprev','insprev']+[f'r{r}' for r in rates[1:]])
        eff = res.loc[[f'r{r}' for r in rates[1:]]]
        print(f"band{bnum} lag{k} n={n}: " + "  ".join(f"{r}:{eff.loc[f'r{r}','b']:.2f}({eff.loc[f'r{r}','se']:.2f})" for r in rates[1:]))
```