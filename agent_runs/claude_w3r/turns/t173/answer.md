The leakage source is identified. At non-drug bundles head 2 reads the e token (weight 0.925) and the empty-u constant (0.075); it never reads slots or z. So the readout only has to be orthogonal to head-2 values of readings-bundle e tokens and to the constant W_V2·b1, while encoding the rate from drug-bundle u.

Next I'll fit that readout in head 2's 48-dim value space with a grouped holdout. The constant is pinned to zero via heavy weight, and I'll report the residual sd at drug bundles and the leakage sd at non-drug bundles. Both ≲ 0.05 rate units would give an essentially clean gated source.

```python
Wv2 = (Wv[96:144]).numpy(); g1n, b1n_ = P['core.prelude.0.norm1.weight'].numpy(), P['core.prelude.0.norm1.bias'].numpy()
Vu = (TOK['u']*g1n + b1n_) @ Wv2.T                      # drug-bundle u values (2406)
Ve = (TOK['e'][~DRG]*g1n + b1n_) @ Wv2.T                # non-drug e values
Ge = GALL[~DRG]; vb1 = b1n_ @ Wv2.T
def fit_c(Y, w, tr_u, tr_e, lam=1e-6):
    M_ = Vu[tr_u].T @ Vu[tr_u] + w*Ve[tr_e].T @ Ve[tr_e] + 1e4*np.outer(vb1, vb1) + lam*np.eye(48)
    return np.linalg.solve(M_, Vu[tr_u].T @ Y[tr_u])
tu = np.array([g not in test for g in Gu]); te_ = np.array([g not in test for g in Ge])
for w in [0.1, 1, 10, 100]:
    for nm, Y in [('ins', Yi), ('ne', Yn)]:
        c_ = fit_c(Y, w, tu, te_); ru = Y[~tu] - Vu[~tu] @ c_; le = Ve[~te_] @ c_
        print(w, nm, 'drug resid sd', ru.std().round(4), '(with drug>0:', ru[Y[~tu]>0].std().round(4), ') leak sd', le.std().round(4), 'leak max', np.abs(le).max().round(3), 'c.vb1', round(float(c_@vb1),5))
```