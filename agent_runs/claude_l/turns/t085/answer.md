The first-step test also fails: the best z error is 1.46. So the problem is not ctx at later steps. Something basic in how the input bundle is embedded, or in what the stored z is, differs from my assumptions.

The embedding has categorical tables I have not included: `action`, `discharged_to`, `route`, `status`, `text`. It also has `kq`, whose role I guessed. If any of them has a non-zero "absent" row 0, the real model adds it to every atom and my atom vectors are off by a constant.

Next I print the row norms of every embedding table, the bin tables' row 0, the `kq` rows and the numeric weights. Zero row-0s (padding) would rule this out and point to how z is stored. Non-zero row-0s mean I should add them and rerun the k=0 search.

```python
for n in NAMES:
    if n.startswith('embed.') and n.endswith('weight') and len(SHAPES[n])==2 and SHAPES[n][1]==192:
        w=W0[n]; print(n, tuple(w.shape), 'row norms:', np.round(w.norm(dim=1).numpy()[:17],3))
print('numeric w col norms', np.round(W0['embed.numeric.weight'].norm(dim=0).numpy(),3), 'bias norm', W0['embed.numeric.bias'].norm().item())
print('z_post.2.bias first/last32 norms', W0['z_post.2.bias'][:32].norm().item(), W0['z_post.2.bias'][32:].norm().item())
print('X z col sd', Xt[:,960:].std(0)[:8].numpy().round(3), 'mean', Xt[:,960:].mean(0)[:8].numpy().round(3))
```