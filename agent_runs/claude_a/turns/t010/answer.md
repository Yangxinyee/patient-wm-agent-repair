Run 9 shows that the insulin multiplier behaves like independent per-record noise:
- Lag-1 autocorrelation is 0.006.
- Between-stay variance is close to what iid noise predicts.
- Correlation with the glucose trend and level is essentially 0.

So within a tier, the rate is effectively randomised. That identifies the insulin dose-response from the records.

This also locates the likely confounding. In the rollout, recorded doses (chosen from the hidden real glucose) are fed in after the model's own sampled readings. A higher rate can then act as a proxy for "real glucose was in a higher tier", which is not a drug effect.

Next I'll estimate the causal effect of the rate on later glucose using only the within-tier variation. Per tier, I'll compare the mean of the next finger and lab glucose across rates, adjusting for the current glucose. If higher rates lower later glucose, insulin works and I can size it. A flat profile would mean no effect.

```python
H = P[P.h>=0][['stay_id','h','glucose_finger','glucose_lab','map_arterial','map_cuff','insulin','norepinephrine']].copy()
# make complete hourly grid 0..24 per stay
grid = pd.MultiIndex.from_product([tr.stay_id.unique(), range(0,25)], names=['stay_id','h'])
H = H.set_index(['stay_id','h']).reindex(grid).reset_index()
H['I'] = H.insulin.fillna(0.0); H['N'] = H.norepinephrine.fillna(0.0)
H['tier'] = np.select([H.glucose_finger>=250, H.glucose_finger>=220, H.glucose_finger>=180, H.glucose_finger>=150],[4,3,2,1], 0)
H.loc[H.glucose_finger.isna(),'tier'] = -1
gg = H.groupby('stay_id')
for k in range(1,7):
    H[f'gf_p{k}'] = gg.glucose_finger.shift(-k); H[f'gl_p{k}'] = gg.glucose_lab.shift(-k)
from numpy.linalg import lstsq
def ols(df, y, X):
    df = df.dropna(subset=[y]+X)
    A = np.column_stack([np.ones(len(df))]+[df[c].values for c in X])
    b, *_ = lstsq(A, df[y].values, rcond=None)
    r = df[y].values - A@b
    s2 = r.var()*len(r)/(len(r)-A.shape[1]); cov = s2*np.linalg.inv(A.T@A)
    return pd.Series(b,['c']+X), pd.Series(np.sqrt(np.diag(cov)),['c']+X), len(df)
for tier in [1,2,3]:
    sub = H[(H.tier==tier)&(H.I>0)]
    for k in [1,2,3]:
        for y in [f'gf_p{k}', f'gl_p{k}']:
            b,se,n = ols(sub, y, ['glucose_finger','I'])
            print(f'tier {tier} lag {k} {y}: coef I {b["I"]:.2f} (se {se["I"]:.2f}) coef G {b["glucose_finger"]:.3f} n={n}')
```