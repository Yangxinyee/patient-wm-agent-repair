Run 92 shows the mismatch is the same at every position, about 0.6 even for the second bundle of a stay, and after every type of previous bundle. So nothing accumulates across steps that the states fail to show. A fixed, step-independent term is wrong. Because each z dimension is still well correlated with the recorded one, I'll redo the "missing constant vector in ctx" fit from run 89, which diverged, now with a stronger ridge penalty and damped steps.

- **If it works:** a modest constant (norm around 10–15) cuts the error well below 0.6. Then the missing piece is essentially the gap term, and I can match the fitted constant against gap(x) to find the input convention.
- **If it fails:** no such constant exists. I'll stop the exact rebuild and design an edit that I verify with the real model through states requests.

```python
def fit_const2(lam, iters=8, step=0.5):
    c = torch.zeros(192)
    for it in range(iters):
        z_, pre_ = zfun(ctx_b + c, EE_)
        g_ = gelu_grad(pre_)
        A = torch.zeros(192,192); bvec = torch.zeros(192)
        for i_ in range(len(z_)):
            J = (W2z * g_[i_][None,:]) @ W1z[:, :192]
            A += J.T @ J; bvec += J.T @ (ZT_[i_] - z_[i_])
        c = c + step*torch.linalg.solve(A + lam*torch.eye(192), bvec - lam*c)
    z_, _ = zfun(ctx_b + c, EE_)
    return c, float(torch.median(torch.norm(z_-ZT_, dim=1)/torch.norm(ZT_, dim=1)))
for lam in [1.0, 10.0, 100.0]:
    c_, er_ = fit_const2(lam)
    xs = np.linspace(-1, 2, 61)
    cs = [float(Fnn.cosine_similarity(lin_(Fnn.gelu(lin_(torch.tensor([float(x_)]), 'gap.0')), 'gap.2'), c_, 0)) for x_ in xs]
    print('lam', lam, 'err', round(er_,4), 'norm c', round(float(torch.norm(c_)),2), 'best cos with gap(x)', round(max(cs),3), 'at x', round(float(xs[int(np.argmax(cs))]),2))
```