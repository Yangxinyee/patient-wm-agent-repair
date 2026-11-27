The first-bundle mismatch remains 0.89 even with the best gap input, so the error is not just the gap convention. Something structural is still off, and the steady 0.75 slope hints at a scaling difference somewhere.

Before going further into z, I'll check pieces I can verify directly from the states.
- **out:** if out = LayerNorm(mean of the slots) with core.out_norm's parameters, then (out − bias)/weight should have mean 0 and SD 1. It should also equal the normalised mean of the four reported h slots.
- **z vs other quantities:** I'll check whether the reported z relates to my candidate through a squashing function, or whether it matches the second half of z_post (log-sd) or the prior.

That will show whether the reported quantities follow the stated formulas, and which parts of my reading are right.

```python
out_t = torch.tensor(X[:, :192]); hs = torch.tensor(X[:, 192:960]).reshape(-1, 4, 192)
u_ = (out_t - Wt['core.out_norm.bias'])/Wt['core.out_norm.weight']
print("out normalized: mean", u_.mean(1)[:5].numpy().round(4), "std", u_.std(1, unbiased=False)[:5].numpy().round(4))
recon = ln(hs.mean(1), 'core.out_norm')
print("out vs out_norm(mean slots h): max err", (recon - out_t).abs().max().item())
for eps in [1e-5, 1e-6]:
    r2 = F.layer_norm(hs.mean(1), (192,), Wt['core.out_norm.weight'], Wt['core.out_norm.bias'], eps); print(eps, (r2-out_t).abs().max().item())
# z true: compare with prior of initial context
act = F.silu
g = mlp(torch.tensor([-0.15]), 'gap', act)
zpr = mlp(Wt['core.out_norm.bias']+g, 'z_prior', act)
print("corr z_true vs prior mean:", np.corrcoef(zpr[:32].numpy(), z_true.numpy())[0,1])
```