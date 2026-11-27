The NE policy is a multiplicative random-walk titration on map_arterial (cuff only when there is no arterial reading). Each hour, rate_t = rate_{t-1} × exp(adj + noise), capped at 1.0, where:
- adj ≈ +0.1 when MAP < 65;
- adj ≈ 0 when MAP is 65-74;
- adj ≈ -0.25 when MAP ≥ 75, with occasional stops;
- the noise has a log-sd of about 0.3.

NE starts at about 0.05 when MAP < 65. Like insulin, the NE rate carries a lot of random variation, here accumulated over hours. Insulin also fits rate = band base × exp(N(0, 0.3)) rounded to 0.5: the predicted share at 0.5 U/h in band 1 is 16.9% and at ≥1.5 it is 22.8%, against 16.9% and 23.1% observed.

Next I'll estimate the 1-hour insulin effect from the within-band randomisation. I'll regress gf_{t+1} − gf_t on rate_t with band fixed effects and controls, then do it separately by band to check whether the effect is linear in dose or scales with glucose. A clear negative slope consistent across bands would mean a real, roughly linear insulin effect. A slope near zero would mean insulin does little in this ICU.

```python
open('/work/notes.md','a').write("""
## NE policy (runs 13-14)
- Uses map_arterial if present else cuff. Start (~0.05) when MAP<65. Titration: log(rate_t/rate_{t-1}) = adj + N(0,~0.3); adj ~ +0.1 (<65), 0 (65-74), -0.25 (>=75, sometimes stop). Cap 1.0.
- So NE rate = accumulated random walk => lots of exogenous variation.
- Insulin rate ~ base(band)*exp(N(0,0.3)) rounded to 0.5 (fits band-1 shares).
""")
gg = g.copy().sort_values(['stay_id','h']).reset_index(drop=True)
gg['ins0'] = gg['ins'].fillna(0); gg['nor0'] = gg['nor'].fillna(0)
for c in ['gf','gl','ma','mc']:
    gg[c+'_ff'] = gg.groupby('stay_id')[c].ffill()
    for k in range(1,7):
        gg[f'{c}_l{k}'] = gg.groupby('stay_id')[c].shift(-k)
gg['glu_now'] = gg.gf.fillna(gg.gl)
def band(v):
    return pd.cut(v, [0,150,180,220,250,300,5000], right=False, labels=False)
gg['band'] = band(gg.glu_now)
d = gg[gg.ins.notna() & gg.gf.notna() & gg.gf_l1.notna()].copy()
d['dg1'] = d.gf_l1 - d.gf
from sklearn.linear_model import LinearRegression
for bnd, dd in d.groupby('band'):
    if len(dd)<200: continue
    X = np.c_[dd.ins, dd.gf, dd.h]
    lr = LinearRegression().fit(X, dd.dg1)
    # also by rate level means
    m = dd.groupby('ins').dg1.agg(['mean','count'])
    print('band', bnd, len(dd), 'slope per U/h', round(lr.coef_[0],3), ' mean dg1 by rate:', {k:(round(r['mean'],1),int(r['count'])) for k,r in m.iterrows() if r['count']>300})
```