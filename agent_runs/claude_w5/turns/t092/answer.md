Run 87 shows that GELU is the best activation everywhere, both in z_post and in the set encoders, yet the error stays at about 0.62. So the documented structure fits as well as anything I have tried, and the mismatch must come from one of the two inputs, ctx or e, being systematically off.

To tell which, I'll use the pattern across the 32 z dimensions, where some match well (correlation about 0.9) and others badly (near 0). From the analytic Jacobian of z_post at the recorded inputs I'll compute how strongly each z dimension responds to changes in ctx versus changes in e, weighted by how much each input varies across bundles. If the badly matched dimensions are the ones driven mainly by e, my bundle encoding is wrong; if they are driven mainly by ctx, the context is wrong.

```python
def gelu_grad(x):
    return 0.5*(1+torch.erf(x/math.sqrt(2))) + x*torch.exp(-0.5*x*x)/math.sqrt(2*math.pi)
W1z = T_['z_post.0.weight']; b1z = T_['z_post.0.bias']; W2z = T_['z_post.2.weight'][:32]
inp = torch.cat([ctx_b, EE_, torch.zeros_like(EE_)], 1)
pre = inp @ W1z.T + b1z
zh = Fnn.gelu(pre) @ W2z.T + T_['z_post.2.bias'][:32]
gg = gelu_grad(pre)
sens_c, sens_e = np.zeros(32), np.zeros(32)
dc = ctx_b - ctx_b.mean(0); de = EE_ - EE_.mean(0)
for i_ in range(len(inp)):
    J = W2z * gg[i_][None, :]
    sens_c += (torch.norm((J @ W1z[:, :192]) @ dc.T, dim=1)**2).numpy()/len(dc)
    sens_e += (torch.norm((J @ W1z[:, 192:384]) @ de.T, dim=1)**2).numpy()/len(de)
pc2 = np.array([np.corrcoef(zh[:,j].numpy(), ZT_[:,j].numpy())[0,1] for j in range(32)])
share_e = sens_e/(sens_e+sens_c)
order_ = np.argsort(pc2)
print('dim corr / e-share (sorted by corr):')
print(np.round(np.c_[pc2[order_], share_e[order_]].T, 2))
print('corr(match, e-share)', np.corrcoef(pc2, share_e)[0,1])
```