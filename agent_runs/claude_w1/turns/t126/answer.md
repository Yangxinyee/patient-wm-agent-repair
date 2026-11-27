The residual regressions give the first direct estimate of what the model is missing. Corrected by the tool-minus-replica coefficients from setA (+0.66, +0.08, +0.03), the results are:
- **Insulin → finger glucose.** The model misses about −2.1, −1.6 and −1.0 mg/dL per u/h of innovation at lags 1–3, about 0 at lag 4, and small positive values (+0.2 to +0.45) at lags 5–10.
- **NE → arterial MAP.** It misses most of the innovation effect: +2.8, +3.7, +3.2, +2.8, +2.2, +1.6, +1.0, +0.7 mmHg per unit at lags 1–8. These include titration feedback, so they understate a sustained doubling.

The NLL constraint is looser than I feared. Finger residual sd is ~14 mg/dL, so a dose-term error with sd 1.4 mg/dL costs only ~0.005 nats per finger reading, or ~0.002 per reading overall. Insulin doses at lags 1–3 are strongly correlated, and the model's own sampled readings carry effects forward. So a head-level edit driven by a lag-1 decoder, with a gain folding in lags 2–3 and calibrated by tool rollouts, could work.

The test is the factual noise of a lag-1 minimum-variance decoder: unit response to a lag-1 insulin dose change, minimum variance on factual states. I'll also check its calibration on the perturbed records. A noise sd of ~0.3 u/h or less supports the head-level route. Around 1 u/h or more, I'd need to amplify the dose signal upstream first.

```python
a1 = A[0]
def mvdr(Sig, cons, resp):
    Si_ = np.linalg.inv(Sig); C = np.stack(cons); Gm_ = C@Si_@C.T
    w_ = Si_@C.T@np.linalg.solve(Gm_, np.array(resp, float)); return w_, float(np.array(resp)@np.linalg.solve(Gm_, np.array(resp, float)))
for lam in [1e-4, 1e-3, 1e-2]:
    Sg = np.cov(xF.T) + lam*np.trace(np.cov(xF.T))/192*np.eye(192)
    w1, nv = mvdr(Sg, [a1], [1.0])
    yv = xF@w1; e1 = (yv - yv.mean()) - (Lnu[:,0] - Lnu[:,0].mean())
    sl = np.polyfit(LdD[:,0], dX@w1, 1)[0]; r2p = np.corrcoef(LdD[:,0], dX@w1)[0,1]**2
    print('ridge %.0e | MVDR noise sd %.3f | factual: sd(w.x) %.3f, err sd vs nu1 %.3f, corr %.3f | pert slope %.3f R2 %.3f' % (lam, np.sqrt(nv), yv.std(), e1.std(), np.corrcoef(yv, Lnu[:,0])[0,1], sl, r2p))
```