The 48-dim head-2 value space is too limited. Once leakage from readings e-tokens is suppressed and the empty-u constant is pinned to zero, the insulin rate error rises to about 0.47; norepinephrine does better (0.27 with leakage 0.02).

The remedy is to give head 2 a new value direction read from the full 192-dim normalised token. That direction would then only need to be orthogonal to readings e-tokens (head 2 never reads slots or z) and to the empty-u constant. Next I'll refit in that space with a grouped holdout. Drug residual ≲ 0.1 with leakage ≲ 0.05 would justify replacing head 2's least-used value direction with this dedicated one.

```python
Xe_nd = TOK['e'][~DRG]; bg = b1n_/g1n
def fit_at(Y, w, tr_u, tr_e, lam=1e-6):
    M_ = TOK['u'][tr_u].T @ TOK['u'][tr_u] + w*Xe_nd[tr_e].T @ Xe_nd[tr_e] + 1e4*np.outer(bg, bg) + lam*np.eye(192)
    return np.linalg.solve(M_, TOK['u'][tr_u].T @ Y[tr_u])
for w in [0.1, 1, 10, 100]:
    for nm, Y in [('ins', Yi), ('ne', Yn)]:
        at_ = fit_at(Y, w, tu, te_); ru = Y[~tu] - TOK['u'][~tu] @ at_; le = Xe_nd[~te_] @ at_
        print(w, nm, 'drug resid sd', ru.std().round(4), '(drug>0', ru[Y[~tu]>0].std().round(4), ') leak sd', le.std().round(4), 'max', np.abs(le).max().round(3), 'a.b1', round(float(at_@bg),5))
```