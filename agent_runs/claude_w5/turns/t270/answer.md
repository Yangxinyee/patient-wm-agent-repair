The fitted u-token corrections at drug bundles j5 to j13 point in almost the same direction (cosines 0.97–0.999). j3 is somewhat different (about 0.7), and none is aligned with u itself (cos about 0.002). They still leave residuals of 8e-5 to 5e-4.

So the tool's prelude attention at drug bundles differs from mine in a fairly consistent way. A ridge fit of u is an indirect probe, though, and its direction may just be the most sensitive one. A more direct test is to fit a common correction δao to the prelude attention output added to the four slot tokens, which are nearly identical. I'll then decompose δao into per-head, per-token value contributions, Wo applied to each token's head value. If δao is reproduced by shifting attention weight between particular tokens in particular heads, for example from u to the e or slot tokens in heads 0 and 3, that pins down a masking or attention-logit convention I can implement. If δao fits poorly, the difference is not in the prelude attention.

```python
print([n for n in NAMES if n.startswith('core.prelude.0.attn')], [SHAPES[n] for n in NAMES if n.startswith('core.prelude.0.attn')])
def msx_ao(d, dao=None, W=None):
    globals()['WCUR'] = W if W is not None else W64
    out0,h0,dt,e,u,Mv,Ma = d['out0'],d['h0'],d['dt'],d['e'],d['u'],d['Mv'],d['Ma']
    hf = h0*torch.exp(-dt*torch.exp(Wt('core.flow.a'))); gp = mlp2_(torch.tensor([math.log(dt)/10.0],dtype=torch.float64),'gap')
    ctx = out0 + LNn(hf.mean(0),'core.out_norm') + gp; z = mlp2_(torch.cat([ctx,e,u]),'z_post')[:32]
    toks = torch.cat([hf+Wt('core.slots'), e[None], u[None], linb(z,'z_in')[None]])
    n1 = LNn(toks,'core.prelude.0.norm1'); ao,att = mha_(n1,n1,'core.prelude.0.attn')
    if dao is not None: ao = ao.clone(); ao[:4] = ao[:4] + dao
    x1 = toks+ao
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
    out=LNn(y.mean(0),'core.out_norm'); return (out,y,z,E,hf,h_new), att, n1
DAO = {}
for j in [3,5,7]:
    dj = prep(j, Xo5); (r0_, att_, n1_) = msx_ao(dj); base = st_vec(r0_); tgt = Xo5[idx[j]]; dl = tgt-base
    J = np.zeros((992,192)); eps_=1e-5
    for i in range(192):
        v = torch.zeros(192,dtype=torch.float64); v[i]=eps_; J[:,i] = (st_vec(msx_ao(dj, v)[0]) - base)/eps_
    for lam in [1e-6, 1e-3]:
        sol = np.linalg.solve(J.T@J+lam*np.eye(192), J.T@dl); nl = st_vec(msx_ao(dj, torch.tensor(sol))[0])
        print(j, 'lam', lam, '|dao|', np.linalg.norm(sol).round(4), 'err', (np.linalg.norm(tgt-nl)/np.linalg.norm(tgt)).round(9))
    DAO[j] = sol
    print(j, 'attn slot0 per head over [s0..s3,e,u,z]:', att_[:,0,:].numpy().round(4))
```