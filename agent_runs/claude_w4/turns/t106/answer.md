The replica is exact (error 1e-6), but this result changes the picture in two ways:
- **No free units:** every one of the 768 hidden units in mlp1 and mlp2 is active for some drug bundle.
- **LayerNorm discards magnitude:** σ(y) before the LayerNorm ranges from 17 to 119 for insulin-only bundles. The insulin rate mostly lives in the magnitude of y, so u_k keeps only its direction. That explains the rate collapse.

My earlier readout test was also unfair. Rates above 11.5 are clipped to identical inputs, so those grid points could never be told apart. I now refit the readouts only on rates the encoder can see. Insulin uses multiples of 0.25 up to 11.5; mixed bundles use a structured grid of NE rates.

For mixed bundles, what matters is whether the readout's change between rate r and its multiple m·r (NE fixed) is close to the true change in rate (m−1)·r. Errors that depend only on the NE rate cancel between the two arms of a dose edit.

```python
rIc = np.arange(0.25, 11.51, 0.25); rI0 = np.r_[0.0, rIc]
rNc = np.r_[0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
Ui_only = np.stack([U_of([ins_atom(float(r))]) for r in rI0])
m_ = Ridge(alpha=1e-10).fit(Ui_only, rI0); e_ = m_.predict(Ui_only) - rI0
print('I-only r<=11.5: rms %.4f max %.4f |w| %.0f' % (np.sqrt((e_**2).mean()), np.abs(e_).max(), np.linalg.norm(m_.coef_)))
sv = np.linalg.svd(Ui_only - Ui_only.mean(0), compute_uv=False); print('I-only curve singular values', np.round(sv[:12],3))
Ucomb = {}; 
for a_ in rI0:
    for b_ in rNc:
        Ucomb[(a_, b_)] = U_of([ins_atom(float(a_)), ne_atom(float(b_))])
K_ = list(Ucomb.keys()); UC = np.stack([Ucomb[k] for k in K_]); aC = np.array([k[0] for k in K_]); bC = np.array([k[1] for k in K_])
m2 = Ridge(alpha=1e-10).fit(UC, aC); pC = m2.predict(UC); e2 = pC - aC
print('I+N combos: rms %.4f max %.4f' % (np.sqrt((e2**2).mean()), np.abs(e2).max()))
sv2 = np.linalg.svd(UC - UC.mean(0), compute_uv=False); print('combo singular values', np.round(sv2[:12],3), 'n>1e-3:', int((sv2>1e-3).sum()))
```