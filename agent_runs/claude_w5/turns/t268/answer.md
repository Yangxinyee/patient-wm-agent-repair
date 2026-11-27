The localisation points downstream of the prelude output, or to a large prelude change. Shifting P alone needs |δP| of about 111, about 7% of P, to cut the error from 2.1e-3 to 3e-4, and a small δP explains little. The attention output at this bundle is identical across slots (|ao| = 231.6 each), as expected when every slot attends almost entirely to the u token.

One candidate is that the prelude's u token differs from the u fed to z_post. That would leave z exact while changing the attention output, by about 2% in n3 as my decoders imply. It would also change the FFN input in a way a δP fit at P cannot reproduce, since P is downstream of the FFN. To test this, I'll let the prelude's u token and e token be overridden independently of z_post, compute Jacobians of the output state with respect to each, and fit.
- If a small change to the prelude u token alone (a few % of |u|) reproduces the tool exactly, the tool feeds the prelude a different action vector. I'll then identify it, for example END atoms included or a missing final norm.
- If not, the discrepancy is in the blocks.

```python
def msx_tok(d, du_tok=None, de_tok=None, W=None):
    globals()['WCUR'] = W if W is not None else W64
    out0,h0,dt,e,u,Mv,Ma = d['out0'],d['h0'],d['dt'],d['e'],d['u'],d['Mv'],d['Ma']
    hf = h0*torch.exp(-dt*torch.exp(Wt('core.flow.a'))); gp = mlp2_(torch.tensor([math.log(dt)/10.0],dtype=torch.float64),'gap')
    ctx = out0 + LNn(hf.mean(0),'core.out_norm') + gp; z = mlp2_(torch.cat([ctx,e,u]),'z_post')[:32]
    ut = u if du_tok is None else u + du_tok; et = e if de_tok is None else e + de_tok
    toks = torch.cat([hf+Wt('core.slots'), et[None], ut[None], linb(z,'z_in')[None]])
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
    out=LNn(y.mean(0),'core.out_norm'); return out,y,z,E,hf,h_new
def fit_tok(j, which):
    dj = prep(j, Xo5); base = st_vec(msx_tok(dj)); tgt = Xo5[idx[j]]; dl = tgt-base; eps_=1e-5
    J = np.zeros((992,192))
    for i in range(192):
        v = torch.zeros(192,dtype=torch.float64); v[i]=eps_
        J[:,i] = (st_vec(msx_tok(dj, **{which: v})) - base)/eps_
    out=[]
    for lam in [1e-4, 1e-2, 1.0]:
        sol = np.linalg.solve(J.T@J+lam*np.eye(192), J.T@dl)
        nl = st_vec(msx_tok(dj, **{which: torch.tensor(sol)}))
        out.append((lam, np.linalg.norm(sol).round(4), (np.linalg.norm(tgt-nl)/np.linalg.norm(tgt)).round(8)))
    return out, sol
for which in ['du_tok','de_tok']:
    for j in [3,5]:
        o, s_ = fit_tok(j, which); print(which, j, 'base err', CMP[(CMP.sid==2020000014)&(CMP.j==j)].err_o.values, o)
        if which=='du_tok' and j==3: SOLU3 = s_
```