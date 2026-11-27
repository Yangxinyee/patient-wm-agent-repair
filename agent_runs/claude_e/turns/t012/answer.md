The within-tier multiplier behaves like fresh noise each hour:
- the correlation between consecutive hours is 0.008;
- the within-stay variance is 0.106 of a total 0.114, so there is essentially no stay-level component;
- its correlations with hour, MAP, lab glucose and previous glucose are all ≤0.05, and the small values likely come from tier mixing.

The run then crashed only because `Q.ne` hit pandas' `.ne` method; I'll use `Q['ne']`.

Working hypothesis: the dose is set by the sliding scale on the recorded finger-stick times an independent random factor. If so, comparing outcomes across that random factor within a tier, holding the current reading fixed, gives the causal effect of a dose on the next reading.

Next I'll regress next-hour finger glucose on the current insulin rate. I'll include fixed effects for the current reading (exact value) and within-tier controls, and run it separately per tier. A clearly negative, tier-stable coefficient would indicate a real lowering effect. A zero would suggest insulin does nothing in this ICU.

```python
from numpy.linalg import lstsq
def fe_reg(df, y, x, fe, extra=[]):
    # regression of y on x (+extra) with fixed effects for 'fe' (demeaning)
    d = df[[y,x,fe]+extra].dropna()
    cols = [y,x]+extra
    dm = d[cols] - d.groupby(fe)[cols].transform('mean')
    X = dm[[x]+extra].values; Y = dm[y].values
    b = lstsq(X, Y, rcond=None)[0]; r = Y - X@b
    se = np.sqrt(np.diag(np.linalg.inv(X.T@X)) * r.var())
    return b, se, len(d)
Pn['dG1'] = Pn.gf_next - Pn.gf_now
Pn['gf_prev'] = Gtr['gf'].shift(1,axis=1).stack(dropna=False).reindex(pd.MultiIndex.from_arrays([Pn.stay,Pn.h])).values if False else np.nan
for tier in ['150-180','180-220','220-300']:
    d = Pn[(Pn.tier==tier)&Pn.on_ins]
    b,se,n = fe_reg(d,'gf_next','ins','gf_now'); print(tier,'gf_next ~ ins | FE(gf_now):', b.round(3), se.round(3), n)
print('NE on? share:', Q['ne'].notna().mean())
```