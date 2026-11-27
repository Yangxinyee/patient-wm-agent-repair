The stored states are self-consistent: `out` equals `out_norm` of the mean of the stored slots, so the layout is as documented. Even the best ctx variant correlates only 0.76 with the stored z, so the problem is not ctx alone.

The most suspicious remaining assumption is e on action-only bundles. I set it to zero. The documentation says only that u is zero when a bundle has no actions. It says nothing about e when a bundle has no measurements. Plausibly e is then the encoding of an empty set: pooled = `size(log 1)`, then mlp2, then the norm, which is a constant vector. A wrong e would distort z on every action bundle. Through the state it would also distort the following reading bundles, which matches the larger error there.

**This run.** I set e on action-only bundles to that empty-set constant and redo the ctx variants. A large drop in the action-only error confirms it. If nothing changes, the next suspects are the measurement encoder (pooling or size term) and the input to z_post.

```python
def empty_enc(pre):
    pooled = F.linear(torch.log1p(torch.zeros(1, 1)), Wt[pre + '.size.weight'], Wt[pre + '.size.bias'])
    return LNt(mlp2l(pooled, pre + '.mlp2', 'gelu'), pre + '.norm')[0]
e_empty = empty_enc('encode'); u_empty = empty_enc('encode_action')
E2 = E_.clone(); E2[~torch.from_numpy(hasmeas)] = e_empty
hasact = np.zeros(nB, bool); hasact[ATM.bidx.values[ATM.ekind.values == 2]] = True
U2 = U_.clone(); U2[~torch.from_numpy(hasact)] = u_empty
on_hf = LNt(hf.mean(1), 'core.out_norm')
for enm, Eb, Ub in [('e0,u0', E_, U_), ('eEmpty,u0', E2, U_), ('e0,uEmpty', E_, U2), ('eEmpty,uEmpty', E2, U2)]:
    for cnm, ctx_ in [('full', out_p + on_hf + gfull), ('nogap', out_p + on_hf), ('noflow,nogap', 2 * out_p)]:
        zm_ = zpost_from(ctx_, Eb[ip_], Ub[ip_])
        e_all = (zm_ - ztrue).abs().mean(1)
        c_ = np.corrcoef(zm_.flatten().numpy(), ztrue.flatten().numpy())[0, 1]
        print(f'{enm:14s} {cnm:13s} all {float(e_all.mean()):.4f}  act-only {float(e_all[act_m].mean()):.4f}  meas-only {float(e_all[~act_m].mean()):.4f} corr {c_:.3f}')
```