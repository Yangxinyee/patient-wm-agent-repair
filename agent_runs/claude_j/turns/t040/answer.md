The factual CSV gives one row per reading: observed value, predicted mean and NLL. I can join the model's predicted glucose_finger at hour h+1 to my panel of reading at h → dose decision at h+0.083 → reading at h+1.

Next I'll regress both the actual and the predicted next reading on the dose surprise. I'll control flexibly for the previous reading, using 10 mg/dL bins plus the band, and limit the sample to readings above 150, where insulin is given. If the predicted-mean slope is well below the actual slope, the model under-uses the exogenous dose signal even within support. The edit then has to strengthen the dose pathway itself, not just its extrapolation.

```python
import statsmodels.api as sm_ if False else None
from numpy.linalg import lstsq
fg = fa_gf[['stay_id','hour','observed','predicted_mean']].copy()
fg = fg[np.isclose(fg.hour, fg.hour.round(0))]; fg['h1'] = fg.hour.round(0).astype(int)
Q = P2.merge(fg[['stay_id','h1','observed','predicted_mean']], on=['stay_id','h1'])
print(len(Q), (Q.observed-Q.g1).abs().max())
Qa = Q[Q.g0>150].copy()
def reg(y, df, extra=[]):
    gb = np.clip((df.g0//10).astype(int), 15, 35)
    X = pd.get_dummies(gb, prefix='gb').astype(float)
    X['g0'] = df.g0.values
    X['dev'] = df.dev.values
    for e in extra: X[e] = df[e].values
    b, *_ = lstsq(X.values, y, rcond=None)
    r = y - X.values@b; s2 = r.var(); cov = s2*np.linalg.inv(X.values.T@X.values)
    i = list(X.columns).index('dev')
    return b[i], np.sqrt(cov[i,i])
for col in ['g1','predicted_mean']:
    print(col, 'dev slope %.3f (se %.3f)' % reg(Qa[col].values, Qa), 'n', len(Qa))
print('model resid on dev: %.3f (se %.3f)' % reg((Qa.g1-Qa.predicted_mean).values, Qa))
```