On the first bundle, putting the current bundle in memory clearly helps: the h error drops from 40 to 9.7 against entries of about 31. So the cross-attention to memory is real, but other details are still wrong. This bundle has no drug atoms, so the open questions are:
- is a zero action token still included in the prelude, or left out?
- is the age embedding added to the keys after projection, before projection, or to both keys and values?

Next I'll run that small grid on bundle 0. If one combination brings the error to about 1e-5, the structure is settled.

```python
def step0v(e_k, u_k, z_k, utok=True, agemode='post', agev=False, memnorm=False):
    h0 = torch.zeros(4,192, dtype=torch.float64)
    tl_ = [h0 + W['core.slots'], e_k[None]] + ([u_k[None]] if utok else []) + [linp(z_k,'z_in')[None]]
    toks = torch.cat(tl_, 0)
    E = lnp(tlayer(toks,'core.prelude.0')[:4], 'core.prelude_norm')
    s = h0 + E
    ages = torch.zeros(1, dtype=torch.float64)
    agek = linp(Fn.gelu(linp(torch.log1p(ages)[:,None], 'core.age.0')), 'core.age.2')
    for p in ['core.blocks.0.0','core.blocks.0.1']:
        n1 = lnp(s, p+'.norm1'); s = s + mha(n1, n1, p+'.attn')
        n2 = lnp(s, p+'.norm2'); Mm = e_k[None]
        if memnorm: Mm = lnp(Mm, p+'.norm2')
        if agev: s = s + mha(n2, Mm + agek, p+'.cross')
        elif agemode=='post': s = s + mha(n2, Mm, p+'.cross', kadd=agek)
        else: s = s + mha(n2, Mm, p+'.cross', kpre=agek)
        s = s + ffn(lnp(s, p+'.norm3'), p)
    sp = Fn.softplus(W['core.delta_in'])
    hn = sp*linp(E,'core.b_in',bias=False) + (s - E)
    y = tlayer(linp(hn,'core.c_out',bias=False), 'core.coda.0')
    return lnp(y.mean(0),'core.out_norm'), y
idx_ = np.where(sid_==st0)[0]; e0, u0 = bvd(BUN[st0][0][1]); z0 = Xt[idx_[0],960:992]
for utok in [True, False]:
    for agemode, agev in [('post',False),('pre',False),('post',True)]:
        for memnorm in [False, True]:
            o_, y_ = step0v(e0,u0,z0,utok,agemode,agev,memnorm)
            print(utok, agemode, agev, memnorm, 'out err', round((o_-Xt[idx_[0],:192]).abs().max().item(),5), 'h err', round((y_.reshape(-1)-Xt[idx_[0],192:960]).abs().max().item(),4))
```