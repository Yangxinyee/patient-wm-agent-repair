Key facts from the architecture:
- **Value head.** It is a plain 16-bin categorical (bin 0 means no value), and the predicted mean is Σ p(b)·centre(b).
- **z handling.** The state update uses the posterior mean of z. At evaluation, the head averages over 4 prior draws of z.
- **Dose prediction.** The model also predicts drug rates with a rate head of the same form.

Since the head input carries almost no dose signal, the workable design is a two-part edit:
1. Amplify the insulin dose dependence of the u-token value in the prelude by a factor A around a fixed reference dose. This is a low-rank edit that leaves empty and NE-only action tokens unchanged.
2. Remove the resulting factual bias with a low-rank edit of the value head's first layer. The edit subtracts (A−1)·Σ a_k·(E[dose_{t−k} | history] − d_ref), so that only the dose innovation is amplified.

Part 2 works only if the policy-expected doses at past lags can be read linearly from the original head input. The test: regress factual doses at lags 1–4, and the response-weighted sum B = Σ h_k·dose_{t−k}, on the head input. Residual sd close to the innovation sd (0.60 per lag, 2.16 for B), with residuals highly correlated with the innovations, means the centring can be nearly exact. A clearly larger residual quantifies the factual cost.

```python
from sklearn.linear_model import Ridge
LD = lagm(Dm, rs, rt, K)
B = LD@hins; T = Lnu@hins
lastbin = np.array([r[3] for r in fsF])[m]
def cvres(Xf, y, alpha=1.0):
    pr = cvp(Ridge(alpha=alpha), Xf, y, cv=5); return y - pr
for k in range(4):
    res = cvres(xF, LD[:,k])
    print('lag%d: sd(D) %.3f resid sd %.3f | sd(nu) %.3f corr(resid,nu) %.3f' % (k+1, LD[:,k].std(), res.std(), Lnu[:,k].std(), np.corrcoef(res, Lnu[:,k])[0,1]))
res = cvres(xF, B)
print('B=sum h_k D: sd %.2f resid sd %.2f | sd(T) %.2f corr(resid,T) %.3f | err sd (resid-T) %.2f' % (B.std(), res.std(), T.std(), np.corrcoef(res, T)[0,1], (res - T + T.mean()).std()))
```