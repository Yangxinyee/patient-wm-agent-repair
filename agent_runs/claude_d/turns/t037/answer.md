No combination matched: the best max error was about 4.2, so something more basic is wrong. Before searching further I want to look at the numbers themselves:
- the target z posterior mean;
- the first and second halves of my z_post output (the mean/log-sd order might be swapped);
- the norms of the stored out and h vectors for the first bundles, to see whether h starts at zero or at the slot embeddings.

Comparing scales and signs should show whether the problem is the output layout, the initial state or the atom features.

```python
an='gelu'; act=F.gelu
e1 = set_enc([atom_vec('hospital_admission', 0, 3)], 'encode', act)
ctx = ln(torch.zeros(192), 'core.out_norm') + mlp(torch.tensor([0.0]), 'gap', act)
zp = mlp(torch.cat([ctx, e1, torch.zeros(192)]), 'z_post', act)
print('target', np.round(target_z1.numpy(),2))
print('mine first32', np.round(zp[:32].numpy(),2))
print('mine last32', np.round(zp[32:].numpy(),2))
print('corr first', np.corrcoef(zp[:32].numpy(), target_z1.numpy())[0,1], 'corr last', np.corrcoef(zp[32:].numpy(), target_z1.numpy())[0,1])
x0 = S['x'][0]; print('out norm', np.linalg.norm(x0[:192]), 'h slot norms', [np.linalg.norm(x0[192+192*s:384+192*s]) for s in range(4)])
print('row1 out norm', np.linalg.norm(S['x'][1][:192]), 'z1', np.round(S['x'][1][960:970],2))
# z posterior across bundles: scale
print('z post mean abs avg per bundle', np.round(np.abs(S['x'][:10,960:]).mean(1),3))
```