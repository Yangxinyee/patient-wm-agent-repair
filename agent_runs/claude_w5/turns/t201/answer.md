The gradient through E is consistent across bundles (median cosine 0.93), but small:
- about 0.55 mg/dL of next-hour glucose per unit of |ΔE|;
- about 0.22 mmHg of MAP per unit of |ΔE|.

Under x2 the model would need next-hour shifts of roughly 4–8 mg/dL or more, which means |ΔE| ≈ 10–15. That is close to the whole norm of E (14) and far outside a linear regime. So the effect cannot be delivered through E alone.

The paths with no size limit are:
- the blocks' FFN output, which is added straight into h_new;
- the coda's FFN output, which is added straight into y.

Their inputs are LayerNormed, so a signal coming from E can only be read there if it falls along directions where those inputs barely vary naturally.

Next I will measure, for both readout locations (block-1 FFN input and coda FFN input):
- the eigen-spectrum of their natural variation over all bundles;
- the best achievable signal-to-variation ratio for a unit E perturbation, using finite-difference Jacobians at 20 drug bundles.

A ratio in the tens or hundreds would make a two-hop gate feasible (prelude gate → E → quiet readout → large write). A ratio near 1 would rule it out.

```python
def cstep_r(out,h,dt_s,e,u,Mv,Mage,dE=None):
    hf=h*torch.exp(-dt_s*torch.exp(W64['core.flow.a']))
    ctx=out+ln_(hf.mean(0),'core.out_norm')+mlp_(torch.tensor([math.log(max(dt_s,1e-9))/10],dtype=torch.float64),'gap')
    zz=mlp_(torch.cat([ctx,e,u]),'z_post')[:32]
    Tk=torch.cat([hf+W64['core.slots'],e[None],u[None],lin_(zz,'z_in')[None]])
    n1=ln_(Tk,'core.prelude.0.norm1'); Tk=Tk+att_(n1,n1,'core.prelude.0.attn')
    Tk=Tk+mlp_(ln_(Tk,'core.prelude.0.norm3'),'core.prelude.0.ffn')
    E=ln_(Tk[:4],'core.prelude_norm')
    if dE is not None: E=E+dE
    xb=hf+E; Mem=Mv+mlp_(torch.log1p(Mage)[:,None],'core.age')
    for l in range(2):
        p='core.blocks.0.%d'%l
        a=ln_(xb,p+'.norm1'); xb=xb+att_(a,a,p+'.attn')
        xb=xb+att_(ln_(xb,p+'.norm2'),Mem,p+'.cross')
        nb=ln_(xb,p+'.norm3'); xb=xb+mlp_(nb,p+'.ffn')
    sp=Fn.softplus(W64['core.delta_in']); abar=torch.exp(-sp*torch.exp(W64['core.a_in']))
    hn=abar*hf+sp*lin_(E,'core.b_in')+(xb-(hf+E))
    x=hn@W64['core.c_out.weight'].T; a=ln_(x,'core.coda.0.norm1'); x2=x+att_(a,a,'core.coda.0.attn')
    nc=ln_(x2,'core.coda.0.norm3'); return nb,nc
NB,NC=[],[]
for d in D[::3]:
    s,k=d['sid'],d['k']; r=RK[s][k]; e,u=enc2(bundles_B[s][k][1]); nb,nc=cstep_r(r['out0'],r['h0'],r['dt'],e,u,r['Mv'],r['Ma']); NB.append(nb); NC.append(nc)
NB=torch.cat(NB); NC=torch.cat(NC)
JB=[];JC=[]
for d in [d for d in D if isdrug(d) and d['sid'] in FIT][::60][:20]:
    s,k=d['sid'],d['k']; r=RK[s][k]; e,u=enc2(bundles_B[s][k][1]); nb0,nc0=cstep_r(r['out0'],r['h0'],r['dt'],e,u,r['Mv'],r['Ma'])
    jb=torch.zeros(4,192,192,dtype=torch.float64); jc=torch.zeros_like(jb)
    for j in range(192):
        dE_=torch.zeros(4,192,dtype=torch.float64); dE_[:,j]=0.1
        nb1,nc1=cstep_r(r['out0'],r['h0'],r['dt'],e,u,r['Mv'],r['Ma'],dE=dE_); jb[:,:,j]=(nb1-nb0)/0.1; jc[:,:,j]=(nc1-nc0)/0.1
    JB.append(jb); JC.append(jc)
for nm,Nn,JJ in [('block1 n3',NB,JB),('coda n3',NC,JC)]:
    Sg=torch.cov(Nn.T); ev_=torch.linalg.eigvalsh(Sg)
    Jm=torch.stack(JJ).reshape(-1,192,192).mean(0)
    M_=Jm.T@torch.linalg.solve(Sg+1e-8*torch.eye(192,dtype=torch.float64),Jm); top=torch.linalg.eigvalsh(M_)[-3:]
    print(nm,'eig min/q05/med/max %.2e %.2e %.2e %.2e'%(float(ev_[0]),float(ev_[9]),float(ev_.median()),float(ev_[-1])),'| SNR^2 per unit |dE| top',np.round(top.numpy(),1))
```