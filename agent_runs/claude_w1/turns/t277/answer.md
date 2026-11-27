Decoding the dose from head 2's 48 value dimensions is only moderate (R² 0.79 for insulin, 0.85 for NE), and the error grows at 2–2.5× doses. That is not clean enough. Decoding from the full 192-d `norm1(u)` should be much better (an earlier note says R² 0.999). In that case I'd repurpose two of head 2's value dimensions so they carry the decoded insulin dose and NE rate exactly, with the matching output-projection columns writing the injection direction. Next I'll fit those decoders, checking accuracy at doubled doses and on empty actions. I'll also rank head 2's value dimensions by their current contribution, to pick the cheapest two to repurpose.

```python
N1_ = ln_(U_, nw_, nb_)
decs = {}
for j_, nm_ in enumerate(['ins', 'ne']):
    A_ = np.c_[N1_, np.ones(len(N1_))]; lam_ = 1e-4
    cdec = np.linalg.solve(A_.T @ A_ + lam_ * np.eye(193), A_.T @ T_[:, j_]); pr_ = A_ @ cdec
    r2_ = 1 - ((pr_ - T_[:, j_])**2).mean() / T_[:, j_].var(); decs[nm_] = cdec
    byf = [np.round(np.sqrt(((pr_ - T_[:, j_])[Fv == f_]**2).mean()), 3) for f_ in [0.0, 0.5, 1.0, 2.0, 2.5]]
    big = T_[:, j_] > np.quantile(T_[:, j_][T_[:, j_] > 0], 0.95)
    print(nm_, "R2 n1", np.round(r2_, 4), "rmse [empty,.5,1,2,2.5]", byf, "intercept", np.round(cdec[-1], 3), "slope at top5%", np.round(np.polyfit(T_[big, j_], pr_[big], 1)[0], 3))
Wout_ = Wo['core.prelude.0.attn.out.weight']
imp = np.linalg.norm(Wout_[:, 96:144], axis=0) * (N1_ @ V2_.T).std(0)
oi = np.argsort(imp); print("head-2 dims least important", oi[:6], np.round(imp[oi[:6]], 4), "median", np.round(np.median(imp), 4))
```