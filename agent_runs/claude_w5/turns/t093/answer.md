Run 88 changes the picture. Without the gap term, every z dimension is in fact well correlated with the recorded one across bundles (0.65–0.93). Yet the relative error per bundle is still about 0.6. High correlation with a large error means a systematic offset: some roughly constant vector is missing from, or wrong in, one of the inputs. The prime suspect is the gap term, which is nearly constant because almost all gaps are 55 or 60 minutes, and which I had evaluated in a way that hurt.

I'll estimate the missing constant directly. By Gauss–Newton on the cached bundles I'll fit one constant vector added to ctx, and separately one added to e, that best reproduces the recorded z. If a constant added to ctx drives the error near zero, I'll then compare it with gap(x) over a range of inputs x, to find the right gap convention.

```python
def zfun(ctx, ee):
    pre_ = torch.cat([ctx, ee, torch.zeros_like(ee)], 1) @ W1z.T + b1z
    return Fnn.gelu(pre_) @ W2z.T + T_['z_post.2.bias'][:32], pre_
def fit_const(which, iters=6, lam=1e-3):
    c = torch.zeros(192)
    blk = slice(0,192) if which == 'ctx' else slice(192,384)
    for it in range(iters):
        ctx_ = ctx_b + (c if which == 'ctx' else 0); ee_ = EE_ + (c if which == 'e' else 0)
        z_, pre_ = zfun(ctx_, ee_)
        g_ = gelu_grad(pre_)
        A = torch.zeros(192,192); bvec = torch.zeros(192)
        for i_ in range(len(z_)):
            J = (W2z * g_[i_][None,:]) @ W1z[:, blk]
            A += J.T @ J; bvec += J.T @ (ZT_[i_] - z_[i_])
        c = c + torch.linalg.solve(A + lam*torch.eye(192), bvec)
    ctx_ = ctx_b + (c if which == 'ctx' else 0); ee_ = EE_ + (c if which == 'e' else 0)
    z_, _ = zfun(ctx_, ee_)
    return c, float(torch.median(torch.norm(z_-ZT_, dim=1)/torch.norm(ZT_, dim=1)))
c_ctx, err_ctx = fit_const('ctx'); c_e, err_e = fit_const('e')
print('err with const in ctx', round(err_ctx,4), 'norm', float(torch.norm(c_ctx)))
print('err with const in e', round(err_e,4), 'norm', float(torch.norm(c_e)))
for x_ in [0.0, 0.3, 0.5, 0.57, 0.7, 0.81, 0.82, 1.0, 1.5, 2.0, 8.19, -1.0]:
    gv = lin_(Fnn.gelu(lin_(torch.tensor([x_]), 'gap.0')), 'gap.2')
    print(x_, 'cos(gap, c_ctx)', round(float(Fnn.cosine_similarity(gv, c_ctx, 0)),3), 'norm', round(float(torch.norm(gv)),2))
```