Encoder variants make no real difference either. Residual connections and pooling choices all sit at about 0.054. The remaining error is not in e's structure.

The `z_post` input weights are largest for the ctx block (norm 12.3, against 7.4 for e and 6.0 for u), so a small ctx error would dominate. A constant offset in ctx, such as a missing bias-like term, would show up as an error that is the same for every stay. A wrong scaling would show up as error that varies with the stay.

So I'll split the z error into its mean over stays (a bias) and the spread around that mean. I'll also try scaled combinations α·out + β·LN(h) + γ·gap to see whether any weighting fits much better. A dominant bias points to a missing constant in ctx. An optimum away from weights of (1, 1, 1) points to a different composition.

```python
zz_best = mlp2l(torch.cat([ctx_b, E1g, torch.zeros_like(E1g)], 1), 'z_post', ACT['relu'])[:, :32]
D_ = (z1_true - zz_best)
print('error: mean over stays |bias| avg %.4f ; residual after bias avg %.4f ; total %.4f' % (D_.mean(0).abs().mean(), (D_ - D_.mean(0)).abs().mean(), D_.abs().mean()))
outp = outS[first].double(); lnh = variants['h_only_flow']; gp = mlp2l(torch.tensor(np.log(dts)/10)[:, None], 'gap', ACT['gelu'])
grid_r = []
for al in [0, 0.5, 1, 1.5, 2]:
    for be in [0, 0.5, 1, 1.5, 2]:
        for ga in [0, 1, 2]:
            ctx_ = al*outp + be*lnh + ga*gp
            zz_ = mlp2l(torch.cat([ctx_, E1g, torch.zeros_like(E1g)], 1), 'z_post', ACT['relu'])[:, :32]
            grid_r.append(((zz_ - z1_true).abs().mean().item(), al, be, ga))
grid_r.sort(); print(grid_r[:6])
# check: is ctx perhaps built from h BEFORE coda?  compare first-state quantities
print('norm out %.2f, LN(h) %.2f, gap %.2f' % (outp.norm(dim=1).mean(), lnh.norm(dim=1).mean(), gp.norm(dim=1).mean()))
```