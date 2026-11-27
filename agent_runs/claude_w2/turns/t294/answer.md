The fit failed only on the μ = 0 case: the intercept is unconstrained there, which makes the system singular. I'll put a tiny ridge on the intercept and run the same scan.

Context from the partial output: the factual glucose residual sd is 14.4, and glucose is about 45% of readings. If MAP were left untouched, glucose rows could absorb about 0.011 nats each, roughly 4.6 (mg/dL)² of extra mean-squared error.

The criterion is unchanged: a μ where the factual MSE barely rises while the arm responses keep at least half the target.

```python
for mu in [0.0, 0.03, 0.1, 0.3, 1.0]:
    A_ = Gd + mu * Gf + np.diag(np.r_[1e-3 * np.ones(192), 1e-6]); w = np.linalg.solve(A_, hd + mu * hf)
    pd_ = Xd @ w; s = Xf @ w
    out = [round(float(np.mean((res_[fte] - lam * s[fte]) ** 2 - res_[fte] ** 2)), 2) for lam in [1.0, 2.0]]
    resp = {a: round(float(pd_[te & (arm_ == a)].mean() / dm[te & (arm_ == a)].mean()), 2) for a in [1, 2, 3, 4, 6, 7]}
    print(f"mu {mu}: dR2 te {r2(dm[te], pd_[te]):.3f} fact sd {s[fte].std():.2f} dMSE(l=1,2) {out} resp/target {resp}")
```