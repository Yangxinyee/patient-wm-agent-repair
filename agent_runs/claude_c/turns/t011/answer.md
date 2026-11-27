The rate noise is essentially iid and exogenous:
- the lag-1 autocorrelation of the log deviation is 0.004;
- the variance of stay means (0.006) is what pure sampling noise predicts, so there is no per-stay factor;
- correlations with glucose within the tier, previous glucose, previous rate, MAP, norepinephrine and hour are all within ±0.02.

The policy is therefore: base rate by finger tier (1/2/3/4 U/h, or the lab tier if no finger-stick), times lognormal noise with a log-sd of about 0.3, rounded to 0.5, plus about 3% random skips. Within a tier the noise acts as a randomised experiment, which identifies insulin's causal effect.

Next I will estimate the one-hour effect. For hour pairs with finger readings at h and h+1, I will regress G(h+1) on the insulin rate at h, with flexible controls for tier and the reading at h. Hour h+1 comes before the next dose, so there is no mediation through later dosing. A clearly negative coefficient means insulin lowers glucose. A coefficient near zero would support "absent", given the noise variation is large.

```python
from numpy.linalg import lstsq
def ols(y, X, names):
    X = np.column_stack(X); b, *_ = lstsq(X, y, rcond=None)
    r = y - X@b; s2 = r@r/(len(y)-X.shape[1]); cov = s2*np.linalg.inv(X.T@X)
    return pd.DataFrame({'b':b,'se':np.sqrt(np.diag(cov))}, index=names)
P['D'] = P.insulin.fillna(0.0)
P['G1'] = P.groupby(level=0).glucose_finger.shift(-1)
q = P[P.glucose_finger.notna() & P.G1.notna()].copy()
q = q[~q.glucose_finger.round().isin([150,180,220,260])]
g = q.glucose_finger.values
cols = [np.ones(len(q)), g, g**2/100, g**3/1e4]
names = ['c','g','g2','g3']
for t in [1,2,3,4]:
    cols.append((q.tf.values==t).astype(float)); names.append(f'tier{t}')
cols.append(q.D.values); names.append('D')
print(ols(q.G1.values, cols, names))
# by tier, separately: effect of D within tier
for t in [1,2,3,4]:
    qt = q[(q.tf==t)]
    gt = qt.glucose_finger.values
    print(t, len(qt), ols(qt.G1.values, [np.ones(len(qt)), gt, qt.D.values], ['c','g','D']).loc['D'].values)
```