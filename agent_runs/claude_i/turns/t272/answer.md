The prelude attention is reproduced exactly: 4 heads of width 48, with keys in the first half of kv and values in the second. The z token's norm is about 2.4. So the earlier result, the same tiny effect at additions of norm 1 to 30, is not plain LayerNorm saturation. It suggests the slots put almost no attention weight on the z token. The constant −0.02 may just be a code-path offset in `step_local`.

Next I'll measure the attention weights directly. For each head, I'll take the mean and spread of the weight that slot queries place on the e, u and z tokens. I'll do this at insulin bundles, norepinephrine bundles and measurement bundles. That tells me which token is a usable carrier, and whether one head already attends strongly and stably to the z or u token.

```python
def attn_w(k):
    hf = HH[k-1]*torch.exp(-dts[k]*torch.exp(fa)) if (k>0 and sid_[k]==sid_[k-1]) else torch.zeros(4,192)
    X = torch.cat([hf+WT['core.slots'], Ee_[k][None], Uu_[k][None], Ztok[k][None]],0); Xn = lnorm(X, WT[P+'norm1.weight'], WT[P+'norm1.bias'])
    return my_attn(Xn[:4], Xn, 4)[1]  # (H,4,7)
kn = np.array([k for k in range(1,NBn) if 'norepinephrine' in INFO[k] and 'insulin' not in INFO[k] and sid_[k]==sid_[k-1]])
km = np.array([k for k in range(1,NBn) if 'glucose_finger' in INFO[k] and sid_[k]==sid_[k-1]])
for nm_, ks_ in [('ins',ki[:300]),('ne',kn[:300]),('meas',km[:300])]:
    A_ = torch.stack([attn_w(k) for k in ks_])  # (n,H,4,7)
    m = A_.mean((0,2)); s = A_.std(0).mean(1)
    print(nm_, 'mean w on [e,u,z] per head:', [[round(float(m[h,j]),3) for j in (4,5,6)] for h in range(4)], ' sd z:', [round(float(s[h,6]),3) for h in range(4)])
```