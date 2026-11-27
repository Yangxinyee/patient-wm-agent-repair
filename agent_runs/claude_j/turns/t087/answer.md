Exact match on bundle 0 (error 0.0). The structure is:
- the action token is always present, as zeros when there are no drug atoms;
- the age embedding is added to the memory vectors before both the key and the value projections;
- the current bundle is in memory at age 0.

For later bundles, two things remain unknown: the unit of age (seconds, minutes or hours) and whether drug-only bundles also put their (empty) e vector into memory. I'll test both on the first 12 bundles of three stays, feeding in the GPU's previous state. The right combination should give errors near 1e-5 at every bundle.

```python
def step(out_prev, h_prev, dt_s, e_k, u_k, z_k, Mem, ages):
    hf = h_prev * torch.exp(-dt_s*torch.exp(W['core.flow.a']))
    toks = torch.cat([hf + W['core.slots'], e_k[None], u_k[None], linp(z_k,'z_in')[None]], 0)
    E = lnp(tlayer(toks,'core.prelude.0')[:4], 'core.prelude_norm')
    s = hf + E
    agek = linp(Fn.gelu(linp(torch.log1p(ages)[:,None], 'core.age.0')), 'core.age.2')
    MA_ = Mem + agek
    for p in ['core.blocks.0.0','core.blocks.0.1']:
        n1 = lnp(s, p+'.norm1'); s = s + mha(n1, n1, p+'.attn')
        n2 = lnp(s, p+'.norm2'); s = s + mha(n2, MA_, p+'.cross')
        s = s + ffn(lnp(s, p+'.norm3'), p)
    sp = Fn.softplus(W['core.delta_in']); abar = torch.exp(-sp*torch.exp(W['core.a_in']))
    hn = abar*hf + sp*linp(E,'core.b_in',bias=False) + (s - (hf+E))
    y = tlayer(linp(hn,'core.c_out',bias=False), 'core.coda.0')
    return lnp(y.mean(0),'core.out_norm'), y
def run_variant2(age_unit, mem_all, kmax=12, stay=None):
    stay = st0 if stay is None else stay
    bl_ = BUN[stay]; idx_ = np.where(sid_==stay)[0]
    Es = [bvd(b[1]) for b in bl_]; times = np.array([b[0] for b in bl_], float)*60.0
    hasm = [any(ev[1] in ('world','start','end') and ev[0] not in ('insulin','norepinephrine') for ev in b[1]) for b in bl_]
    errs = []
    for k in range(1, min(kmax, len(bl_))):
        out_p = Xt[idx_[k-1], :192]; h_p = Xt[idx_[k-1], 192:960].view(4,192); z_k = Xt[idx_[k], 960:992]
        js = [j for j in range(0, k+1) if (mem_all or hasm[j] or j==k)][-128:]
        Mem = torch.stack([Es[j][0] for j in js]); ages = torch.tensor((times[k]-times[js])/age_unit, dtype=torch.float64)
        o_, y_ = step(out_p, h_p, float(times[k]-times[k-1]), Es[k][0], Es[k][1], z_k, Mem, ages)
        errs.append(max((o_-Xt[idx_[k],:192]).abs().max().item(), (y_.reshape(-1)-Xt[idx_[k],192:960]).abs().max().item()))
    return np.round(errs, 5)
for au in [1.0, 60.0, 3600.0]:
    for ma in [True, False]:
        print(au, ma, [run_variant2(au, ma, stay=s_).max() for s_ in list(BUN.keys())[:3]], run_variant2(au, ma)[:8])
```