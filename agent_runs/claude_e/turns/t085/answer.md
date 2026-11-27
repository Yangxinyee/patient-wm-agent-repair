Adding more glucose history changes the true and model coefficients separately, but their difference γ barely moves:

| Specification | γ, lags 1–6 |
|---|---|
| Base controls | [−1.51, −1.25, −0.41, 0.32, 0.70, 0.73] |
| Extra glucose history | [−1.64, −1.25, −0.45, 0.27, 0.75, 0.70] |

That stability is what I expect if the model's residual is orthogonal to the glucose history and only the dose effect is missing. So γ ≈ [−1.6, −1.25, −0.43, 0.3, 0.72, 0.72] is a robust target. Separately, the model also underweights the lab reading (0.23 versus a true 0.47).

Next I need to know whether the head's input q encodes this target. I will build D_t = Σ_k γ_k·dev_{t−k} for every glucose reading and fit a ridge readout from q with folds split by stay. I will also measure how well each individual lag is decodable. A high cross-validated R² (above about 0.7) means a head-level edit can implement the correction; a low one means I need richer features or a different route.

```python
gam = np.array([-1.6, -1.25, -0.43, 0.3, 0.72, 0.72])
def q_mean_rows(Xs, R_, nz=16, seed=0):
    Q = q_from_rows(Xs, torch.as_tensor(R_.prev.values), torch.as_tensor(R_.dt.values, dtype=torch.float64), nz=nz, seed=seed)
    return Q.mean(0).numpy()
def add_dev_lags(R_, G, nlag=10):
    gf = G['gf']; ins = G['ins'].fillna(0)
    hrs_ = np.array(gf.columns); idx = {h:i for i,h in enumerate(hrs_)}
    out = np.zeros((len(R_), nlag))
    gfv = gf.loc[R_.stay.values].values; insv = ins.loc[R_.stay.values].values
    tr = tier(gfv); ed = np.where(tr>=0, EDOSE[np.clip(tr,0,4)], EMISS); dev = insv - ed
    ii = np.array([idx.get(h, -99) for h in R_.hour.values])
    for j in range(nlag):
        c = ii-1-j; ok = c>=0
        out[ok, j] = dev[np.where(ok)[0], c[ok]]
    return out
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
def cv_r2(F, y, groups, alpha=1.0, w=None):
    pred = np.zeros_like(y)
    for tr_, te_ in GroupKFold(5).split(F, y, groups):
        m = Ridge(alpha=alpha).fit(F[tr_], y[tr_], sample_weight=None if w is None else w[tr_]); pred[te_] = m.predict(F[te_])
    return 1 - np.mean((y-pred)**2)/np.var(y), pred
gA = RA.event=='glucose_finger'; g1m = R1.event=='glucose_finger'
QA = q_mean_rows(XA, RA[gA]); Q1 = q_mean_rows(X1, R1[g1m])
LA = add_dev_lags(RA[gA], Gtr); L1 = add_dev_lags(R1[g1m], Gtu)
Fq = np.vstack([QA, Q1]); Lg = np.vstack([LA, L1]); grp = np.concatenate([RA[gA].stay.values, R1[g1m].stay.values])
Dt = Lg[:, :6] @ gam
for al in [0.1, 1, 10]:
    r2, _ = cv_r2(Fq, Dt, grp, al); print('alpha', al, 'CV R2 of D_t from q:', round(r2,3), ' sd(D_t)', Dt.std().round(3))
print('per-lag CV R2:', [round(cv_r2(Fq, Lg[:,j], grp, 1.0)[0],2) for j in range(10)])
```