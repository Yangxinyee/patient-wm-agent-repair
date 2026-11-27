The actual z values are small (mean |z| 0.12) and my predictions correlate only 0.73 with them, so a structural piece is still missing. The ablations give two hints: dropping the kq row slightly improves the correlation, so it is probably not part of the atom vector, and the gap term looks mis-specified.

Before tuning further, I'll check an identity that should hold exactly if I've read the state layout and out_norm correctly: after each bundle, out = core.out_norm(mean over slots of h). If it holds, the layout and LayerNorm (and its epsilon) are right, and the error lies in ctx/gap or the atom/encoder details. If it fails, my picture of the state itself is wrong.

```python
o = X[:, :192]; hh = X[:, 192:960].reshape(-1,4,192)
for eps in [1e-5, 1e-6]:
    pred_o = Fnn.layer_norm(hh.mean(1), (192,), W['core.out_norm.weight'], W['core.out_norm.bias'], eps)
    print(eps, 'max |out - out_norm(mean h)|', (pred_o - o).abs().max().item(), 'mean', (pred_o-o).abs().mean().item())
# also check other combos: maybe out uses the 'y' before something
print('out row norms', o.norm(dim=1)[:5], 'h slot norms', hh.norm(dim=2)[:3])
```