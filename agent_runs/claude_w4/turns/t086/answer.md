The z mismatch (4.27) is my own mistake. `load()` returned the probe edit of z_post, which I never removed, not the original weights. The replicas of the encoder, ctx and END atoms used only unedited tensors, so they are unaffected.

I'll drop all five probe edits and recompute the original z_post at drug bundles. The check is the same:
- z mean should match fwd_3's z posterior to float precision, using the first 32 outputs;
- the unit-importance numbers should then show the real distribution, not the probe's mostly-zero units.

```python
for nm in list(edited()): drop_edit(nm)
print('edited now:', edited())
P3 = XIN3 @ load('z_post.0.weight').T + load('z_post.0.bias'); G3 = gelu(P3)
ZO3 = G3 @ load('z_post.2.weight').T + load('z_post.2.bias')
print('z mean match first half:', float((ZO3[:, :32] - torch.tensor(Z3[dr3])).abs().max()), ' second half:', float((ZO3[:, 32:] - torch.tensor(Z3[dr3])).abs().max()))
W2m = load('z_post.2.weight')[:32]
imp = (G3.std(0) * W2m.norm(dim=0)).numpy(); act_mean = G3.mean(0).numpy()
order = np.argsort(imp)
print('lowest-importance units at drug bundles:', order[:10], imp[order[:10]].round(5))
print('their mean activation:', act_mean[order[:10]].round(3), ' W2m col norms:', W2m.norm(dim=0).numpy()[order[:10]].round(3))
print('importance median', np.median(imp).round(4), 'max', imp.max().round(4))
```