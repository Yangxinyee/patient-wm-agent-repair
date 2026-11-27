The time-flow rates match the documentation, so the decay is right. The gate `a_bar` is only 0.27–0.47, and `c_out` is far from the identity. The state after the very first bundle already has entries around 30. The first bundle is the cleanest test case: the previous state is presumably zero, so neither the time flow nor the previous state can be the source of error. Next I'll run my step on bundle 0 of several stays with h = out = 0, both with the current bundle in memory and with an empty memory. A small error would put the problem in the memory or age handling for later bundles. A large one would mean the step's structure itself (tokens, update or coda) is wrong.

```python
def step0(e_k, u_k, z_k, mem_mode, mode='post'):
    h0 = torch.zeros(4,192, dtype=torch.float64)
    toks = torch.cat([h0 + W['core.slots'], e_k[None], u_k[None], linp(z_k,'z_in')[None]], 0)
    E = lnp(tlayer(toks,'core.prelude.0')[:4], 'core.prelude_norm')
    s = h0 + E
    for p in ['core.blocks.0.0','core.blocks.0.1']:
        n1 = lnp(s, p+'.norm1'); s = s + mha(n1, n1, p+'.attn')
        if mem_mode=='cur':
            ages = torch.zeros(1, dtype=torch.float64)
            agek = linp(Fn.gelu(linp(torch.log1p(ages)[:,None], 'core.age.0')), 'core.age.2')
            n2 = lnp(s, p+'.norm2'); s = s + mha(n2, e_k[None], p+'.cross', kadd=agek)
        s = s + ffn(lnp(s, p+'.norm3'), p)
    sp = Fn.softplus(W['core.delta_in'])
    hn = sp*linp(E,'core.b_in',bias=False) + (s - E)
    y = tlayer(linp(hn,'core.c_out',bias=False), 'core.coda.0')
    return lnp(y.mean(0),'core.out_norm'), y, hn
for stay in list(BUN.keys())[:3]:
    idx_ = np.where(sid_==stay)[0]; e0, u0 = bvd(BUN[stay][0][1]); z0 = Xt[idx_[0],960:992]
    for mm in ['cur','none']:
        o_, y_, hn_ = step0(e0, u0, z0, mm)
        print(stay, mm, 'out err', round((o_-Xt[idx_[0],:192]).abs().max().item(),5), 'h err', round((y_.reshape(-1)-Xt[idx_[0],192:960]).abs().max().item(),4), '|y| mean', round(y_.abs().mean().item(),2))
```