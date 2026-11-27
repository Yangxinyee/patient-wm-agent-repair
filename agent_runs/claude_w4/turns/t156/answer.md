The replica is far off. The error in h is about 220–280 against a typical change of 573, and the error in out is about 1.6. The memory variants barely differ, so the problem is structural, not the approximate measurement e vectors.

The time flow and ctx were verified exactly earlier, so h_tf is right. Suspects:
- norm placement in the recurrent blocks and coda (the docs say only the prelude is pre-norm);
- my prelude E;
- the state layout.

First the cheapest check: does out_norm(mean over slots of the GPU h) reproduce the GPU out? If yes, the layout is right and the error is inside the block or coda. Then I'll try post-norm variants of blocks and coda.

```python
o_chk = LNw(yt.mean(1), 'core.out_norm')
print('out from gpu y err', (o_chk - ot).norm(dim=1).mean().item(), '|out|', ot.norm(dim=1).mean().item())
def step_var(rws, E, htf, block_pre=True, coda_pre=True, incl=True, unit=3600.0, pre=True):
    Mm, Ag, Mk = build_mem(rws, incl, unit); ae = age_emb(Ag)
    x = htf + E
    for p in ['core.blocks.0.0.', 'core.blocks.0.1.']:
        if block_pre:
            n1 = LNw(x, p+'norm1'); x = x + mha_m(n1, n1, Wc[p+'attn.q.weight'], Wc[p+'attn.kv.weight'], Wc[p+'attn.out.weight'])
            n2 = LNw(x, p+'norm2'); x = x + mha_m(n2, Mm, Wc[p+'cross.q.weight'], Wc[p+'cross.kv.weight'], Wc[p+'cross.out.weight'], kadd=None if pre else ae, mask=Mk, kpre=ae if pre else None)
            x = x + ffn(LNw(x, p+'norm3'), p)
        else:
            x = LNw(x + mha_m(x, x, Wc[p+'attn.q.weight'], Wc[p+'attn.kv.weight'], Wc[p+'attn.out.weight']), p+'norm1')
            x = LNw(x + mha_m(x, Mm, Wc[p+'cross.q.weight'], Wc[p+'cross.kv.weight'], Wc[p+'cross.out.weight'], kadd=None if pre else ae, mask=Mk, kpre=ae if pre else None), p+'norm2')
            x = LNw(x + ffn(x, p), p+'norm3')
    sp = torch.nn.functional.softplus(Wc['core.delta_in']); abar = torch.exp(-sp*torch.exp(Wc['core.a_in']))
    hn = abar*htf + sp*(E @ Wc['core.b_in.weight'].T) + (x - (htf + E))
    c = hn @ Wc['core.c_out.weight'].T
    if coda_pre:
        n1 = LNw(c, 'core.coda.0.norm1'); c = c + mha_m(n1, n1, Wc['core.coda.0.attn.q.weight'], Wc['core.coda.0.attn.kv.weight'], Wc['core.coda.0.attn.out.weight'])
        c = c + ffn(LNw(c, 'core.coda.0.norm3'), 'core.coda.0.')
    else:
        c = LNw(c + mha_m(c, c, Wc['core.coda.0.attn.q.weight'], Wc['core.coda.0.attn.kv.weight'], Wc['core.coda.0.attn.out.weight']), 'core.coda.0.norm1')
        c = LNw(c + ffn(c, 'core.coda.0.'), 'core.coda.0.norm3')
    return c, hn
for bp in [True, False]:
    for cp in [True, False]:
        y_, hn_ = step_var(rows, E0, htf, bp, cp)
        print('block_pre', bp, 'coda_pre', cp, 'err y', round((y_-yt).norm(dim=(1,2)).mean().item(),2))
```