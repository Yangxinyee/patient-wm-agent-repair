The activation was the problem. My earlier "relu" detection was a false match: it hit the word "prelude". With GELU in both the set encoder and `z_post`, my posterior correlates 0.58 with the saved one, the first variant to beat the prior's 0.50.

So the MLPs are almost certainly GELU throughout, including the gap, `z_prior` and value-head MLPs I rebuilt earlier with ReLU. That would explain the 3-unit bias in the head rebuild, and it means my earlier list of "dead" value-head units was computed wrongly.

This run switches every MLP to GELU and re-checks both rebuilds:
1. the head's predicted means with the plain last_numeric layout and no free fitting;
2. the posterior correlation.

If GELU is right, the head error should fall well below the 3 units seen before without fitting anything, and the posterior correlation should rise further. Remaining posterior error would then point to the unit-table mapping or the exact z_post input order, which I can re-search cheaply.

```python
def mlp(x, p): return lin(gelu(lin(x, p + '.0')), p + '.2')
def set_enc(vecs, p):
    V_ = np.array(vecs); n_ = len(V_); s_ = mlp(V_, p + '.mlp1').sum(0) / np.sqrt(n_)
    return ln(mlp(s_ + lin(np.array([np.log1p(n_)]), p + '.size'), p + '.mlp2'), W[p + '.norm.weight'], W[p + '.norm.bias'])
pm_pred_B = predict3(feats_fn=FF['B']); report(pm_pred_B, 'GELU, layout B')
for nm_ in ['h,a,z,0,0,0,0,0', 'h-a-z-h-0-0-0-1?', 'h,a,z,h,0,1,0,1']: report(predict3(feats_fn=FF[nm_]), 'GELU ' + nm_)
pre_c = [(ctx_of(ip, dt), [base_mk(e_, v_) for e_, v_ in zip(g_.event, g_.value)], X[idx_, 960:992]) for idx_, ip, dt, g_ in cand2]
for order in ['ceu', 'cue']:
    P_ = np.array([mlp(np.concatenate([{'c': c_, 'e': set_enc(v_, 'encode'), 'u': np.zeros(192)}[k_] for k_ in order]), 'z_post')[:32] for c_, v_, t_ in pre_c])
    print(order, 'post corr %.4f' % np.mean([np.corrcoef(P_[:, j], T_all[:, j])[0, 1] for j in range(32)]), 'mean abs err %.4f' % np.abs(P_ - T_all).mean())
Pp = np.array([mlp(c_, 'z_prior')[:32] for c_, v_, t_ in pre_c]); print('prior corr %.4f' % np.mean([np.corrcoef(Pp[:, j], T_all[:, j])[0, 1] for j in range(32)]))
```