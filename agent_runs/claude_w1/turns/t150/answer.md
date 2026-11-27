Run 139 changes the plan in three ways.
- **The real model's residuals are not linearly predictable from the head input** (CV R² ≈ −0.05 for glucose, −0.03 for MAP). The 18% predictability in my replica's residuals was an artefact of that replica.
- **So any correction that is a function of the head input is pure factual cost.** It cannot pick up the residual's link to the true innovations, because that information is missing from the head input.
  - This also explains the model's weak dose response: given a noisy internal dose signal, a small response is what maximum likelihood picks.
  - The factual cost of my edit is therefore about Var(correction)/(2σ²) per reading.
- **The true residuals still load on rule innovations,** e.g. glucose [−0.7, −2.1, −0.5]. That fits the explanation above.

To match the true effects with an affordable factual cost, the correction's factual variance must be close to the floor set by the true innovations. That needs much cleaner dose decoding.

Next I'll compute minimum-variance decoders, a constrained regression:
- the dose-response directions come from my factual-versus-perturbed head-input pairs (`dX` on `LdD`);
- the decoders must have unit response to the lagged doses;
- they minimise factual variance, with the register's bin allowed as a free offset.

I'll compare their factual variance with the ridge decoders and with the irreducible innovation variance.

How to read the outcome:
- Factual variance close to the innovation variance (about 0.25 for insulin): a correctly sized correction could fit within the budget.
- Variance far above it: I'll need an upstream change or a narrower correction.

```python
fm_ = (codesK == 'glucose_finger') & (hrK >= 1)
Xf_ = Xk[fm_]; prevf_ = np.array([t[3] for t in keep], dtype=float)[fm_]
lbf_ = np.array([lastbin_of('glucose_finger', v) if np.isfinite(v) else 0 for v in prevf_])
A_ = np.c_[LdD[:, :6], np.ones(len(LdD))]
Dc_, *_ = np.linalg.lstsq(A_, dX, rcond=None); D_ = Dc_[:6]
r2_ = 1 - ((dX - A_ @ Dc_)**2).sum() / ((dX - dX.mean(0))**2).sum()
print('response linear fit R2', round(r2_, 3), 'norms', np.linalg.norm(D_, axis=1).round(3), 'lastbins', np.unique(lbf_).tolist())
Xr_ = Xf_.copy()
for bb_ in np.unique(lbf_):
    mk_ = lbf_ == bb_; Xr_[mk_] -= Xf_[mk_].mean(0)
Sig_ = np.cov(Xr_.T); Si_ = np.linalg.inv(Sig_ + 1e-4 * np.trace(Sig_) / 192 * np.eye(192))
Lnf_ = lagmat(NUm, ixK[fm_], hrK[fm_], 6)
wr_ = DEC['glucose_finger'].coef_
print('ridge decoders: response D@w (rows lag1-3 of D)', (D_[:3] @ wr_.T).round(2).tolist(), 'factual var within-bin', np.var(Xr_ @ wr_.T, axis=0).round(3))
M_ = D_ @ Si_ @ D_.T
for c_ in [np.array([1., 0, 0, 0, 0, 0]), np.array([-2.08, -0.79, -0.27, 0, 0, 0]), np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])]:
    u_ = Si_ @ D_.T @ np.linalg.solve(M_, c_)
    lb_ = np.var(Lnf_ @ c_)
    print(c_.round(2).tolist(), 'MVDR factual var %.3f' % np.var(Xr_ @ u_), 'innovation floor %.3f' % lb_, 'resp', (D_ @ u_).round(2).tolist(), '|u| %.2f' % np.linalg.norm(u_))
```