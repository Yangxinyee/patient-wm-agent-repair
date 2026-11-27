The plain and within-cell ridge decoders predict factual lag-1 doses well (error sd about 0.53 u/h). However, they barely respond to the edits: 0.13 against a needed 1.64 for ×2. They must be reading dose through policy-linked glucose features rather than through any effect of the dose itself, which means `V` holds the unperturbed samples. The perturbed-dose sets are probably `Vp`/`Lp_` or `spX`/`spD`. Next I'll check which pairs actually differ from the recorded doses, fit ridge decoders on them, and score them the same way. A decoder trained on perturbed doses should show a real edit response (well above 0.5 out of 1.64).

```python
dfac = lambda a_, b_: np.round(np.linalg.norm(a_ - b_, axis=1).mean(), 3)
print("V-Vp", dfac(V, Vp), "V-spX", dfac(V, spX), "Vp-spX", dfac(Vp, spX))
print("corr F3~Lp0", np.round(np.corrcoef(F[:,3], Lp_[:,0])[0,1],3), "F3~spD0", np.round(np.corrcoef(F[:,3], spD[:,0])[0,1],3),
      "frac equal", np.round((F[:,3]==Lp_[:,0]).mean(),3), np.round((F[:,3]==spD[:,0]).mean(),3))
for nm_, X_, y_ in [('Vp/Lp', Vp, Lp_[:,0]), ('spX/spD', spX, spD[:,0]), ('Vp/spD', Vp, spD[:,0])]:
    A_, t_y = X_[fm], y_[fm]; mA_, my_ = A_.mean(0), t_y.mean()
    S_ = (A_ - mA_).T @ (A_ - mA_); c_ = (A_ - mA_).T @ (t_y - my_)
    for lam in [1e1, 1e2]:
        w_ = np.linalg.solve(S_ + lam * np.eye(192), c_); b_ = (t_y - A_ @ w_).mean()
        print(nm_, lam, score(w_, b_))
```