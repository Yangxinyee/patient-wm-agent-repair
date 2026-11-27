Still no match: the best candidate is off by 1.6, while z values run to about ±3.

Before trying more combinations blindly, I'll measure how close the best candidate is element by element (correlation and regression slope against the true z). A high correlation would mean the structure is right and a small term (for example the gap input or one embedding row) is off. A low correlation would mean something structural is wrong, such as the encoder form or z_post's input.

I'll also check whether e_1 itself looks plausible. Bundle 2's z depends on e_2 and the evolved ctx, so it isn't a direct test of the encoder.

```python
act = F.silu
e1 = set_encode(atom_vec(11,0,3,use_kq=False)[None], 'encode', act)
g = mlp(torch.tensor([0.0]), 'gap', act)
zz = mlp(torch.cat([Wt['core.out_norm.bias']+g, e1, torch.zeros(192)]), 'z_post', act)[:32]
print(np.corrcoef(zz.numpy(), z_true.numpy())[0,1], np.polyfit(zz.numpy(), z_true.numpy(), 1))
print(np.c_[zz.numpy()[:10], z_true.numpy()[:10]].round(3))
# sensitivity: how much does z change if e1 replaced with zeros?
z_noe = mlp(torch.cat([Wt['core.out_norm.bias']+g, torch.zeros(192), torch.zeros(192)]), 'z_post', act)[:32]
print("corr with e=0 :", np.corrcoef(z_noe.numpy(), z_true.numpy())[0,1])
```