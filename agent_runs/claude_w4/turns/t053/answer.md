The copy is badly off, and in the part that should be easiest. Across stays z_1 varies by only 0.074 on average per dimension, so it is set almost entirely by the shared context. My errors (mean 0.32) are several times that variation, so the context or z-posterior formula is wrong, not the per-reading encoding.

A common latent-variable setup that would explain this: the posterior is written as a correction on the prior, posterior mean = prior mean(ctx) + z_post output. I will test that, with a few context variants (h before or after the time decay, gap or no gap), on bundles 0 and 1. If one variant brings both errors near zero, the copy is back on track. If not, I will drop the full copy and switch to measuring the model through the tool.

```python
def zpost_variants(ctx, E, act):
    n = E.shape[0]; inp = torch.cat([ctx.expand(n,-1), E, torch.zeros(n,192)],1)
    zp = lin(act(lin(inp,'z_post.0')),'z_post.2')
    pr = lin(act(lin(ctx,'z_prior.0')),'z_prior.2')
    return {'post_raw': zp[:,:32], 'prior+post': pr[...,:32] + zp[:,:32], 'prior_only': pr[...,:32].expand(n,-1)}
E1g = {}
for an in ['gelu','relu']:
    act = acts[an]
    E1g[an] = torch.stack([set_enc2(torch.stack([meas_atom(ev, v, dict(kq=0), {'mmHg':5,'mg/dL':1}) for ev, v in b]), 'encode', act) for b in b1])
rows_ = []
for an in ['gelu','relu']:
    act = acts[an]
    for hv in ['flowed','raw']:
        hh = h0f if hv=='flowed' else h0
        for gv in ['/10', 'none', 'raw']:
            gterm = 0 if gv=='none' else mlp(torch.tensor([np.log(3600.0)/(10 if gv=='/10' else 1)],dtype=torch.float32),'gap',act)
            for ov in ['out+norm', 'out', 'norm']:
                ctx = {'out+norm': o0 + ln(hh.mean(0),'core.out_norm'), 'out': o0, 'norm': ln(hh.mean(0),'core.out_norm')}[ov] + gterm
                for k_, zz in zpost_variants(ctx[None], E1g[an], act).items():
                    err = (zz - Z1).abs()
                    rows_.append((err.mean().item(), err.max().item(), an, hv, gv, ov, k_))
rows_.sort(key=lambda t: t[0])
for r_ in rows_[:8]: print(r_)
```