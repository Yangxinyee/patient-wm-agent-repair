Every ctx variant gives nearly the same error, about 0.056–0.059. Even dropping the gap term entirely barely matters (0.0588). So the remaining error does not come from ctx details. It comes from a larger structural element: the bundle encoding e, or how the posterior mean is formed from the `z_post` output.

To locate it I'll compare my computed posterior with the true z dimension by dimension (slope, intercept, correlation). I'll also compare the prior mean, and check how much the error moves if e is zeroed:
- High correlation but slopes away from 1 means a scaling or parameterisation issue, such as the mean being combined with the prior.
- Low correlation means the encoding itself is wrong.
- If the prior alone tracks z well, the posterior is a correction on top of the prior.

```python
ctx_b = variants['full_flow'] + mlp2l(torch.tensor(np.log(dts)/10)[:, None], 'gap', ACT['gelu'])
zm = mlp2l(torch.cat([ctx_b, E1g, torch.zeros_like(E1g)], 1), 'z_post', ACT['relu'])
zmean, zls = zm[:, :32], zm[:, 32:]
zpr = mlp2l(ctx_b, 'z_prior', ACT['relu']); zpm, zpls = zpr[:, :32], zpr[:, 32:]
zt_ = z1_true
def dimstats(a_, b_):
    out_ = []
    for d_ in range(32):
        x_, y_ = a_[:, d_].numpy(), b_[:, d_].numpy()
        c_ = np.corrcoef(x_, y_)[0, 1]; sl = np.polyfit(x_, y_, 1)
        out_.append((round(c_, 3), round(sl[0], 3), round(sl[1], 3)))
    return out_
st_post = dimstats(zmean, zt_); st_prior = dimstats(zpm, zt_)
print('post: corr/slope/int per dim', st_post[:12]); print('prior:', st_prior[:12])
print('median corr post %.3f prior %.3f' % (np.median([s[0] for s in st_post]), np.median([s[0] for s in st_prior])))
zm0 = mlp2l(torch.cat([ctx_b, torch.zeros_like(E1g), torch.zeros_like(E1g)], 1), 'z_post', ACT['relu'])[:, :32]
print('err with e=0: %.4f ; err post %.4f ; err prior %.4f' % ((zm0 - zt_).abs().mean(), (zmean - zt_).abs().mean(), (zpm - zt_).abs().mean()))
# precision-weighted combination test
vp, vq = torch.exp(2*zpls), torch.exp(2*zls)
comb = (zpm / vp + zmean / vq) / (1/vp + 1/vq)
print('precision-combined err %.4f' % (comb - zt_).abs().mean())
print('z_true std per dim (first 8)', np.round(zt_.std(0).numpy()[:8], 3), ' err per dim (first 8)', np.round((zmean - zt_).abs().mean(0).numpy()[:8], 3))
```