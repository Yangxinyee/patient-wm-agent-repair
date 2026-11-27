out_norm over the GPU slots reproduces the GPU out exactly, so the state layout is right. Pre-norm for both blocks and coda is clearly the right structure (the post-norm variants are 7–12 times worse). A 38% error remains, though.

To localize it, I'll measure the cross-attention's contribution and the error when it is removed. If removing it barely changes the error, the fault is elsewhere (prelude E or the update formula). If cross-attention output is large, its approximate measurement e vectors are the likely cause. I'll also check whether the error grows with position in the stay, as it would if memory errors accumulate.

```python
def step_parts(rws, E, htf, incl=True, unit=3600.0, pre=True, cross_on=True):
    Mm, Ag, Mk = build_mem(rws, incl, unit); ae = age_emb(Ag)
    x = htf + E; cn = []
    for p in ['core.blocks.0.0.', 'core.blocks.0.1.']:
        n1 = LNw(x, p+'norm1'); x = x + mha_m(n1, n1, Wc[p+'attn.q.weight'], Wc[p+'attn.kv.weight'], Wc[p+'attn.out.weight'])
        n2 = LNw(x, p+'norm2'); co = mha_m(n2, Mm, Wc[p+'cross.q.weight'], Wc[p+'cross.kv.weight'], Wc[p+'cross.out.weight'], mask=Mk, kpre=ae if pre else None, kadd=None if pre else ae)
        cn.append(co.norm(dim=(1,2)).mean().item())
        if cross_on: x = x + co
        x = x + ffn(LNw(x, p+'norm3'), p)
    sp = torch.nn.functional.softplus(Wc['core.delta_in']); abar = torch.exp(-sp*torch.exp(Wc['core.a_in']))
    hn = abar*htf + sp*(E @ Wc['core.b_in.weight'].T) + (x - (htf + E))
    c = hn @ Wc['core.c_out.weight'].T
    n1 = LNw(c, 'core.coda.0.norm1'); c = c + mha_m(n1, n1, Wc['core.coda.0.attn.q.weight'], Wc['core.coda.0.attn.kv.weight'], Wc['core.coda.0.attn.out.weight'])
    c = c + ffn(LNw(c, 'core.coda.0.norm3'), 'core.coda.0.')
    return c, cn
y1, cn1 = step_parts(rows, E0, htf)
y2, _ = step_parts(rows, E0, htf, cross_on=False)
e1 = (y1-yt).norm(dim=(1,2)); e2 = (y2-yt).norm(dim=(1,2))
print('cross norms', np.round(cn1,2), 'err with', round(e1.mean().item(),1), 'without', round(e2.mean().item(),1))
pos = np.array([np.sum(st[:i] == st[i]) for i in rows])
for a, b in [(0,10),(10,25),(25,40),(40,80)]:
    m = (pos >= a) & (pos < b)
    print('pos', a, b, int(m.sum()), round(e1[torch.tensor(m)].mean().item(),1))
```