This scan settles the trade-off for the "do no harm" readout:
- factual noise added by the readout costs about sd²/(2·201) nats on glucose rows, so the budget allows only about 2 mg/dL of it;
- per unit of that noise, the readout gives only about 0.3 of the open-loop response;
- even at μ = 0.3 and gain 1, glucose bin NLL rises +0.011 for an x2 response of −0.77 against a one-step target of −2.38.

Matching the target this way is impossible within the fit budget.

The only way to buy budget is if the readout's factual part genuinely predicts the model's residual. Earlier that looked like a mean-squared-error gain of −20, but it was measured against stored probabilities averaged over z draws. That comparison may be partly artificial, since the readout uses the prior-mean z.

I'll redo it consistently:
- residual = recorded value − predicted mean (prior-mean z);
- joint fit, bounded tilt;
- scan μ and gain for bin NLL change and the x2 and x0 responses.

A real gain shows up as negative glucose ΔNLL at gain about 1 with a nonzero response. If no setting gives that, the head-only route is exhausted.

```python
Q_res = np.where(np.isfinite(Q_val), Q_val - Q_pm0, 0.0)
Q_hf = Q_Xa[Q_fr].T @ Q_res[Q_fr] / Q_fr.sum()
print("resid sd (zmu) %.2f mean %.2f" % (Q_res[Q_fa].std(), Q_res[Q_fa].mean()))
for mu in [0.1, 0.3, 1.0]:
    w1 = np.linalg.solve(Q_Gd + mu*Q_Gf + np.diag(np.r_[1e-3*np.ones(192), 1e-6]), Q_hd + mu*Q_hf)
    sF = Q_Xa @ w1; te_f = Q_fa & (Q_si >= 50)
    line = f"mu {mu}: fact sd {sF[Q_fa].std():.2f} R2res(test) {1 - ((Q_res[te_f]-sF[te_f])**2).mean()/Q_res[te_f].var():.3f}"
    for lam in [0.5, 1.0, 1.5]:
        PN = Q_P(*Q_build(w1, lam)); pmN = Q_pm(PN)[0]
        dn_ = -np.log(PN[Q_r, Q_bidx] + 1e-12) + np.log(Q_P0[Q_r, Q_bidx] + 1e-12)
        RAV = (pmN[Q_i1] - pmN[Q_i0]) - (Q_pm0[Q_i1] - Q_pm0[Q_i0])
        line += f" | L{lam}: dNLL g {dn_[Q_f0 & Q_gl & (Q_si >= 50)].mean():.4f} (tr {dn_[Q_f0 & Q_gl & (Q_si < 50)].mean():.4f}) x2 {RAV[Q_TE & (Q_ARM == 1)].mean():.2f} x0 {RAV[Q_TE & (Q_ARM == 3)].mean():.2f}"
    print(line)
```