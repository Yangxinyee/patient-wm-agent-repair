Result:
- Dose changes are partly decodable from the head input's arm differences: R² 0.57 on held-out stays (lag 1 0.59, lags 2–4 0.39–0.47).
- The decoded response is attenuated (49–77% of the target) and not linear: x3 gives 57% and x0 49%.
- The decisive problem: applied to factual rows, the readout varies with sd 48 mg/dL against a needed 3.2. Used as is, it would wreck the factual fit.

So the readout must be fitted jointly:
- on factual rows it should predict the model's own residual (recorded value minus predicted mean). This also penalises adding any spread to factual predictions;
- on arm differences it should give the missing dose effect.

I'll scan the factual weight μ and report three things:
- the predicted factual mean-squared-error change. Roughly 2 (mg/dL)² of extra error would cost the 0.005-nat budget on glucose rows;
- the counterfactual response per arm;
- the R² of the arm differences.

Viable means a μ where the factual MSE does not increase while the arm responses keep at least half the target, so a gain of 1.5–2 can calibrate them.

```python
cenA = np.asarray(CEN_of(np.asarray(D4['code'])), dtype=float)
P_ = np.asarray(D4['P'], dtype=float)
if 'pm0' in D4 and len(D4['pm0']) == len(X): pm_ = np.asarray(D4['pm0'], dtype=float)
else: pm_ = (P_[:, 1:] * cenA[:, :16]).sum(1) / P_[:, 1:].sum(1)
val_ = np.asarray(D4['val'], dtype=float); res_ = val_ - pm_
fa = (armD == 0) & gl & np.isfinite(res_)
ftr, fte = fa & (siD < 50), fa & (siD >= 50)
print("factual resid: n %d mean %.2f sd %.2f" % (fa.sum(), res_[fa].mean(), res_[fa].std()))
Xf = np.c_[X, np.ones(len(X))]; Xd = np.c_[dX, np.zeros(len(dX))]
Gd = Xd[tr].T @ Xd[tr] / tr.sum(); hd = Xd[tr].T @ dm[tr] / tr.sum()
Gf = Xf[ftr].T @ Xf[ftr] / ftr.sum(); hf = Xf[ftr].T @ res_[ftr] / ftr.sum()
arm_ = kd.arm.values
for mu in [0.0, 0.03, 0.1, 0.3, 1.0]:
    A_ = Gd + mu * Gf + 1e-3 * np.diag(np.r_[np.ones(192), 0.0]); w = np.linalg.solve(A_, hd + mu * hf)
    pd_ = Xd @ w; s = Xf @ w
    out = []
    for lam in [1.0, 2.0]:
        dmse = np.mean((res_[fte] - lam * s[fte]) ** 2 - res_[fte] ** 2); out.append(round(float(dmse), 2))
    resp = {a: round(float(pd_[te & (arm_ == a)].mean() / dm[te & (arm_ == a)].mean()), 2) for a in [1, 2, 3, 4, 6, 7]}
    print(f"mu {mu}: dR2 te {r2(dm[te], pd_[te]):.3f} fact sd {s[fte].std():.2f} dMSE(l=1,2) {out} resp/target {resp}")
```