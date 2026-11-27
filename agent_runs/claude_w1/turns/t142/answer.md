Run 132 gave the missing direct effects from the residual regression. Standard errors are about 0.12 for glucose and 0.17–0.21 for MAP.
- **Insulin → glucose:** c ≈ [−1.49, −1.53, −1.00] at lags 1–3, then small positive values at lags 5–8 (about +1.2 in total).
- **Norepinephrine → MAP:** R ≈ [2.55, 3.69, 3.22, 2.80, 2.20, 1.60, …], long and large.
  - The norepinephrine rule changes the rate multiplicatively, so an innovation persists into later doses (persistence about 0.73, 0.56, 0.44, …).
  - Removing that persistence leaves per-dose missing effects of about [2.55, 1.83, 0.45, 0.32].
  - This matters for the rollout: there the dose path is fixed, so I need per-dose effects, not the reduced form.
  - For norepinephrine the correction must also be in dose levels, compensated through readings only. Subtracting a rule expectation that is proportional to the previous dose would cancel the response to a ×2 scaling of the whole path.

One worry: if the head is already near-optimal given its input x, a correction that is a function of x alone cannot improve the factual fit. It only adds variance, and the factual-fit limit is 0.005 nats.

To test this, I'll train dose decoders for lags 1–3 on the perturbed-dose samples and apply them to the factual head inputs. Then I'll check whether the residuals load on the decoded dose minus the rule expectation. I'll also check how much residual variance any linear function of x explains.

How to read the outcome:
- Coefficients near c with low decoder noise: the correction gains on factual fit while restoring the dose response.
- Coefficients near zero: the edit has a pure variance cost, and I must keep its factual variance small.

```python
from sklearn.linear_model import Ridge
Vp = np.stack([t[8] for t in SAMP]).astype(np.float64); cp_ = np.array([t[0] for t in SAMP]); sp_ = np.array([t[1] for t in SAMP])
Lp_ = np.array([[t[5], t[6], t[7]] for t in SAMP], dtype=float)
Xk = np.stack([t[8] for t in keep]).astype(np.float64)
DEC = {}
for cd_, NU_, D_ in [('glucose_finger', NUm, Dm), ('map_arterial', NUNm, Rm)]:
    mp = cp_ == cd_
    # CV by stay
    us = np.unique(sp_[mp]); fold = {sv: i % 5 for i, sv in enumerate(us)}; fk = np.array([fold[sv] for sv in sp_[mp]])
    pr_ = np.zeros((mp.sum(), 3))
    for f_ in range(5):
        tr_, te_ = fk != f_, fk == f_
        rm_ = Ridge(alpha=1.0).fit(Vp[mp][tr_], Lp_[mp][tr_]); pr_[te_] = rm_.predict(Vp[mp][te_])
    r2_ = 1 - ((pr_ - Lp_[mp])**2).mean(0) / Lp_[mp].var(0)
    rm_ = Ridge(alpha=1.0).fit(Vp[mp], Lp_[mp]); DEC[cd_] = rm_
    sel = (codesK == cd_) & (hrK >= 1)
    dh_ = rm_.predict(Xk[sel])                      # decoded doses (factual)
    Ld_ = lagmat(D_, ixK[sel], hrK[sel], 3); Ln_ = lagmat(NU_, ixK[sel], hrK[sel], 3); Pi_ = Ld_ - Ln_
    print(cd_, 'CV R2 pert', r2_.round(3), 'resid sd pert', np.sqrt(((pr_ - Lp_[mp])**2).mean(0)).round(3))
    print('   factual: decode err sd', (dh_ - Ld_).std(0).round(3), 'slope dh on d', [np.polyfit(Ld_[:,k], dh_[:,k], 1)[0].round(3) for k in range(3)])
    y0 = res_all[sel]
    b1, s1 = ols_cl(y0, dh_ - Pi_, ixK[sel]); print('   res ~ (dhat - pi):', b1[1:].round(3), s1[1:].round(3))
    b2, s2 = ols_cl(y0, np.c_[dh_, Pi_], ixK[sel]); print('   res ~ dhat, pi   :', b2[1:].round(3), s2[1:].round(3))
    b3, s3 = ols_cl(y0, Ln_, ixK[sel]); print('   res ~ nu (check)  :', b3[1:].round(3))
    # head optimality: CV R2 of res on X
    us2 = np.unique(ixK[sel]); fold2 = {sv: i % 5 for i, sv in enumerate(us2)}; fk2 = np.array([fold2[sv] for sv in ixK[sel]])
    pr2_ = np.zeros(sel.sum())
    for f_ in range(5):
        tr_, te_ = fk2 != f_, fk2 == f_
        pr2_[te_] = Ridge(alpha=10.0).fit(Xk[sel][tr_], y0[tr_]).predict(Xk[sel][te_])
    print('   CV R2 res~X', (1 - ((y0 - pr2_)**2).mean() / y0.var()).round(4), 'res var', y0.var().round(2))
```