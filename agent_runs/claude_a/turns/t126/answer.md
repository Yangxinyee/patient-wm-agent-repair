The reconstruction is exact (max error 8e-7). The prelude tokens are [h after time flow + slot embeddings, e, u, z_in(posterior mean)], keys and values are the two halves of `kv`, and heads are contiguous 48-dimension blocks. So I can compute every head's output per slot, and find which linear read-out through one row of the prelude output projection gives the cleanest write.

**Next step:** regress the insulin write target (+rate at insulin bundles, −rbar(tier) at glucose bundles, 0 elsewhere) on the per-slot head outputs, using held-out stays. Readout sets:
- head 3 alone (rate route);
- heads 2+3;
- all four heads, as the upper bound for a single out-projection row.

I will report residual sd by bundle type, separately for rate-only and baseline targets.

**What would confirm or refute the clean design:**
- Rate residual ≲ 0.1 and baseline residual ≲ 0.15 confirm it.
- If the baseline stays at ~0.3, I will do the baseline differently, e.g. move it to the readout side using the head's last-value register.

```python
with torch.no_grad():
    Hh = (awr @ Vh)                       # nb x 4h x 4q x 48
Hs = Hh.permute(0,2,1,3).reshape(nb_*4, 4*48).numpy().astype(np.float64)   # per (bundle,slot): 192 = heads concat
rep = lambda a: np.repeat(a, 4)
yRate = yR.copy(); yBase = np.where(btype=='gread', yI, 0.0); yBoth = yRate + yBase
trs, tes = rep(tr_m), rep(te_m); bts = rep(btype)
def fit_eval(cols, y, name):
    X = np.c_[Hs[:, cols], np.ones(len(Hs))]
    rg = Ridge(alpha=1e-4, fit_intercept=False).fit(X[trs], rep(y)[trs]); res = rep(y) - rg.predict(X)
    out = f"{name}: " + " ".join(f"{t}:sd{res[tes&(bts==t)].std():.3f}/b{res[tes&(bts==t)].mean():+.3f}" for t in ['ins','gread','other'])
    print(out); return rg
h3 = list(range(144,192)); h2 = list(range(96,144)); allh = list(range(192))
_ = fit_eval(h3, yRate, "rate | head3")
_ = fit_eval(allh, yRate, "rate | all heads")
_ = fit_eval(h2, yBase, "base | head2")
_ = fit_eval(h2+h3, yBase, "base | heads2+3")
_ = fit_eval(allh, yBase, "base | all heads")
_ = fit_eval(allh, yBoth, "both | all heads")
```