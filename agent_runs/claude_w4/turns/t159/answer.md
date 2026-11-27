The replica captures about 87% of the variance in the true state change (fit slope 0.87–0.94). The remaining 13% is still large. Two more facts:
- c_out is far from the identity.
- The coda alone changes the state a lot.

So the structure is roughly right, but one component is wrong. The cross-attention is the prime suspect, since removing it helped slightly. I'll try a few memory treatments in one run:
- M normalized by the block's norm2;
- age added to both keys and values;
- no age at all.

If one drops the error sharply (say below 100), the memory structure is found. Otherwise the problem lies elsewhere.

```python
def step_cv(rws, E, htf, mode):
    Mm, Ag, Mk = build_mem(rws, True, 3600.0); ae = age_emb(Ag)
    x = htf + E
    for p in ['core.blocks.0.0.', 'core.blocks.0.1.']:
        n1 = LNw(x, p+'norm1'); x = x + mha_m(n1, n1, Wc[p+'attn.q.weight'], Wc[p+'attn.kv.weight'], Wc[p+'attn.out.weight'])
        n2 = LNw(x, p+'norm2')
        if mode == 'lnM': mem = LNw(Mm, p+'norm2'); co = mha_m(n2, mem, Wc[p+'cross.q.weight'], Wc[p+'cross.kv.weight'], Wc[p+'cross.out.weight'], mask=Mk, kpre=ae)
        elif mode == 'ageKV': co = mha_m(n2, Mm + ae, Wc[p+'cross.q.weight'], Wc[p+'cross.kv.weight'], Wc[p+'cross.out.weight'], mask=Mk)
        elif mode == 'noage': co = mha_m(n2, Mm, Wc[p+'cross.q.weight'], Wc[p+'cross.kv.weight'], Wc[p+'cross.out.weight'], mask=Mk)
        elif mode == 'lnM_ageKV': mem = LNw(Mm + ae, p+'norm2'); co = mha_m(n2, mem, Wc[p+'cross.q.weight'], Wc[p+'cross.kv.weight'], Wc[p+'cross.out.weight'], mask=Mk)
        x = x + co
        x = x + ffn(LNw(x, p+'norm3'), p)
    sp = torch.nn.functional.softplus(Wc['core.delta_in']); abar = torch.exp(-sp*torch.exp(Wc['core.a_in']))
    hn = abar*htf + sp*(E @ Wc['core.b_in.weight'].T) + (x - (htf + E))
    return coda_only(hn)
for mode in ['lnM', 'ageKV', 'noage', 'lnM_ageKV']:
    y_ = step_cv(rows, E0, htf, mode)
    print(mode, round((y_-yt).norm(dim=(1,2)).mean().item(),1), fitstat(yt-htf, y_-htf))
```