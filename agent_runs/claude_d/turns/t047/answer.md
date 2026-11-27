EPS is the random dose multiplier (m minus its mean), BASE is the policy's step dose, W holds the measurement IPW weights, and EPSN is the NE log-noise. Since m is random given the history, Z = INS − E[INS | BASE] is a clean instrument.

To test linearity, I'll estimate the lag-1 effect of Z on next-hour glucose separately for each base level. This lag is free of any feedback through later doses. The base levels correspond to glucose bands of roughly 165 to 330. A constant effect per unit across levels means a linear, additive effect. An effect that grows with base level or glucose means a proportional effect. I'll also test a quadratic in Z and sign asymmetry. For NE, I'll compare a fit linear in rate against a fit linear in log-rate, by rate level.

```python
from numpy.linalg import lstsq
def ols(y, X):
    X1 = np.column_stack([np.ones(len(y)), X]); b = lstsq(X1, y, rcond=None)[0]
    r = y - X1@b; s2 = r@r/(len(y)-X1.shape[1]); cov = s2*np.linalg.inv(X1.T@X1)
    return b, np.sqrt(np.diag(cov))
rows=[]
N,Hh = INS.shape
for t in range(1,23):
    ok = (INS[:,t]>0)&(BASE[:,t]>0)&~np.isnan(GF[:,t+1])&~np.isnan(Gdec[:,t])
    idx = np.where(ok)[0]
    rows.append(np.column_stack([idx, np.full(len(idx),t), BASE[idx,t], INS[idx,t], Gdec[idx,t], GF[idx,t+1],
                                 INS[idx,t-1], np.nan_to_num(GF[idx,t+2] if t+2<Hh else np.full(len(idx),np.nan), nan=np.nan)]))
R = np.vstack(rows)
bl, insv, gd, y1, insm1 = R[:,2], R[:,3], R[:,4], R[:,5], R[:,6]
Z = np.zeros(len(R))
for b_ in np.unique(bl):
    s_ = bl==b_; Z[s_] = insv[s_] - insv[s_].mean()
print('n', len(R))
for b_ in np.unique(bl):
    s_ = bl==b_
    b, se = ols(y1[s_], np.column_stack([Z[s_], gd[s_], insm1[s_]]))
    print('base %.0f n=%6d Gmean=%.0f  lag1 effect/u = %.2f (%.2f)' % (b_, s_.sum(), gd[s_].mean(), b[1], se[1]))
b, se = ols(y1, np.column_stack([Z, Z**2, Z*(gd-200)/100, gd, insm1, bl]))
print('pooled: Z %.2f(%.2f)  Z^2 %.2f(%.2f)  Z*G/100 %.2f(%.2f)' % (b[1],se[1],b[2],se[2],b[3],se[3]))
b, se = ols(y1, np.column_stack([Z*(Z>0), Z*(Z<=0), gd, insm1, bl]))
print('pos Z %.2f(%.2f)  neg Z %.2f(%.2f)' % (b[1],se[1],b[2],se[2]))
```