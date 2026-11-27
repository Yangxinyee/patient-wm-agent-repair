I now have `mystep`'s exact structure from its bytecode. It follows ARCHITECTURE.md, with one detail: in the cross-attention to memory the age embedding is added to the memory vectors before the key/value projection.

Three things are specific to drug bundles and can be tested:
- (a) whether a drug-only bundle's empty e enters the memory M at all;
- (b) whether the empty e token takes part in the prelude attention;
- (c) whether the u token is used as is.

I'll write `msx`, a copy of `mystep` with switches for these variants. First I'll check that it reproduces `mystep` exactly, then evaluate each variant as a single step from the tool's own previous state at drug bundles j3, j5, j7 and j9 of stay 2020000014.
- A variant whose error drops to about 1e-6 at all four bundles identifies the tool's convention.
- If none does, the cause lies elsewhere, for example the age embedding applied to keys only.

```python
F_ = torch.nn.functional
def msx(d, mem_excl_self=False, drop_e_tok=False, drop_u_tok=False, W=None):
    globals()['WCUR'] = W if W is not None else W64
    out0,h0,dt,e,u,Mv,Ma = d['out0'],d['h0'],d['dt'],d['e'],d['u'],d['Mv'],d['Ma']
    hf = h0*torch.exp(-dt*torch.exp(Wt('core.flow.a')))
    gp = mlp2_(torch.tensor([math.log(dt)/10.0],dtype=torch.float64),'gap')
    ctx = out0 + LNn(hf.mean(0),'core.out_norm') + gp
    z = mlp2_(torch.cat([ctx,e,u]),'z_post')[:32]
    tl = [hf+Wt('core.slots')]
    if not drop_e_tok: tl.append(e[None])
    if not drop_u_tok: tl.append(u[None])
    tl.append(linb(z,'z_in')[None]); toks = torch.cat(tl)
    n1 = LNn(toks,'core.prelude.0.norm1'); ao,att = mha_(n1,n1,'core.prelude.0.attn'); x1 = toks+ao
    n3 = LNn(x1,'core.prelude.0.norm3'); P = (x1 + linb(F_.gelu(linb(n3,'core.prelude.0.ffn.0')),'core.prelude.0.ffn.2'))[:4]
    E = LNn(P,'core.prelude_norm'); xb = hf+E
    if mem_excl_self: Mv, Ma = Mv[:-1], Ma[:-1]
    Mk = Mv + mlp2_(torch.log1p(Ma)[:,None],'core.age')
    for l in (0,1):
        p=f'core.blocks.0.{l}'
        n=LNn(xb,p+'.norm1'); a_,_=mha_(n,n,p+'.attn'); xb=xb+a_
        n=LNn(xb,p+'.norm2'); c_,_=mha_(n,Mk,p+'.cross'); xb=xb+c_
        n=LNn(xb,p+'.norm3'); xb=xb+linb(F_.gelu(linb(n,p+'.ffn.0')),p+'.ffn.2')
    sp_d=F_.softplus(Wt('core.delta_in')); ab=torch.exp(-sp_d*torch.exp(Wt('core.a_in')))
    h_new = ab*hf + sp_d*(E@Wt('core.b_in.weight').T) + (xb-(hf+E))
    c = h_new@Wt('core.c_out.weight').T
    n=LNn(c,'core.coda.0.norm1'); a_,_=mha_(n,n,'core.coda.0.attn'); c=c+a_
    n=LNn(c,'core.coda.0.norm3'); y=c+linb(F_.gelu(linb(n,'core.coda.0.ffn.0')),'core.coda.0.ffn.2')
    out=LNn(y.mean(0),'core.out_norm'); return out,y,z,E,hf,h_new
r_a = msx(d0); r_b = mystep(d0); print('msx vs mystep', float((r_a[1]-r_b[1]).abs().max()))
for var in [dict(), dict(mem_excl_self=True), dict(drop_e_tok=True), dict(mem_excl_self=True, drop_e_tok=True)]:
    errs=[]
    for j in [3,5,7,9]:
        dj = prep(j, Xo5); rj = st_vec(msx(dj, **var)); errs.append(np.linalg.norm(Xo5[idx[j]]-rj)/np.linalg.norm(rj))
    print(var, np.round(errs,7))
```