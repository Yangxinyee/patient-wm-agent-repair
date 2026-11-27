The least-squares fit to u is meaningless. Explaining the discrepancy needed a δu of norm 14,000 against |u| of 12, and the nonlinear check failed, so no small change in u explains it. That fits z being exact. I've also ruled out the memory and e-token conventions. The step structure is the same for every bundle, yet the mismatch appears only at drug bundles.

One explanation left is numerical. The tool runs float32 on GPU, and some part of the drug-bundle computation, where attention saturates on u and the FFN input mixes a huge state with drug information, may be ill-conditioned enough that float32 departs from my float64 by about 1e-3. That is cheap to test: I'll run the same step at j3–j9 in float32 with float32 weights.
- If the float32 local result matches the tool to about 1e-6, precision is the explanation. I must then simulate in float32 and make the edit robust to it.
- If float32 matches float64 instead, there is a structural difference I haven't found.

```python
WF32 = {k: v.float() for k,v in W64.items()}
def msx_dt(d, W, dtype):
    globals()['WCUR'] = W
    cv = lambda v: v.to(dtype) if torch.is_tensor(v) else v
    out0,h0,e,u,Mv,Ma = [cv(d[k]) for k in ['out0','h0','e','u','Mv','Ma']]; dt=d['dt']
    hf = h0*torch.exp(-dt*torch.exp(Wt('core.flow.a')))
    gp = mlp2_(torch.tensor([math.log(dt)/10.0],dtype=dtype),'gap')
    ctx = out0 + LNn(hf.mean(0),'core.out_norm') + gp
    z = mlp2_(torch.cat([ctx,e,u]),'z_post')[:32]
    toks = torch.cat([hf+Wt('core.slots'), e[None], u[None], linb(z,'z_in')[None]])
    n1 = LNn(toks,'core.prelude.0.norm1'); ao,att = mha_(n1,n1,'core.prelude.0.attn'); x1 = toks+ao
    n3 = LNn(x1,'core.prelude.0.norm3'); P = (x1 + linb(F_.gelu(linb(n3,'core.prelude.0.ffn.0')),'core.prelude.0.ffn.2'))[:4]
    E = LNn(P,'core.prelude_norm'); xb = hf+E
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
    out=LNn(y.mean(0),'core.out_norm'); return out.double(),y.double(),z.double(),E.double(),hf.double(),h_new.double()
try:
    for j in [3,4,5,7,9]:
        dj = prep(j, Xo5)
        r64 = st_vec(msx_dt(dj, W64, torch.float64)); r32 = st_vec(msx_dt(dj, WF32, torch.float32)); tgt = Xo5[idx[j]]
        print(j, 'f64 err', np.linalg.norm(tgt-r64)/np.linalg.norm(tgt), 'f32 err', np.linalg.norm(tgt-r32)/np.linalg.norm(tgt), 'f32 vs f64', np.linalg.norm(r32-r64)/np.linalg.norm(r64))
except Exception as ex:
    import traceback; traceback.print_exc()
globals()['WCUR'] = W64
```