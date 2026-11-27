The prelude FFN hidden units decode the targets better than E does (held-out R² about 0.58 for both drugs), but the readout still leaks at reading bundles.

The reason is structural. The target is a product: "a drug is present" × (dose minus expected dose given the state). A single linear readout cannot form that product without leaking.

**Idea:** build the product explicitly with GELU gating in the prelude FFN, where the dose signal is strong. I would repurpose a few dead or near-dead hidden units:
- Each unit's pre-activation = a large drug-present discriminant + the linear target, so GELU is about 0 at non-drug bundles and about linear at drug bundles.
- A twin unit carrying only the discriminant cancels the offset.

**Three checks:**
1. Is the target linearly decodable from the FFN's input (norm3 of the prelude residual) when only drug bundles are used?
2. Do drug and non-drug bundles separate linearly with a wide margin?
3. Are there dead units in the prelude FFN?

```python
t0=time.time()
def n3_of(r,at):
    rec={}; e,u=enc2(at); Mv=r['Mv'].clone(); Mv[-1]=e
    hf=r['h0']*torch.exp(-r['dt']*torch.exp(W64['core.flow.a']))
    ctx=r['out0']+ln_(hf.mean(0),'core.out_norm')+mlp_(torch.tensor([math.log(max(r['dt'],1e-9))/10],dtype=torch.float64),'gap')
    zz=mlp_(torch.cat([ctx,e,u]),'z_post')[:32]
    Tk=torch.cat([hf+W64['core.slots'],e[None],u[None],lin_(zz,'z_in')[None]])
    n1=ln_(Tk,'core.prelude.0.norm1'); Tk=Tk+att_(n1,n1,'core.prelude.0.attn')
    return ln_(Tk,'core.prelude.0.norm3')[:4].float()
for d in D:
    s,k=d['sid'],d['k']; r=RK[s][k]; at=bundles_B[s][k][1]
    d['n3']=n3_of(r,at)
    if 'hid_i2' in d:
        for nm,(cd,f) in VN.items(): d['n3_'+nm]=n3_of(r,scale(at,cd,f))
print('n3 done %.0fs'%(time.time()-t0))
isdrug=lambda d:'hid_i2' in d
rowsX,rowsY,rowsF=[],[],[]
for d in D:
    if not isdrug(d): continue
    ri,rn,ci,cn=d['ri'],d['rn'],d['cI'],d['cN']
    for key,a_,b_ in [('n3',ri-ci,rn-cn),('n3_i2',2*ri-ci,rn-cn),('n3_ih',.5*ri-ci,rn-cn),('n3_n2',ri-ci,2*rn-cn),('n3_nh',ri-ci,.5*rn-cn)]:
        rowsX.append(d[key]); rowsY+=[[a_,b_]]*4; rowsF+=[d['sid'] in FIT]*4
X3=torch.cat(rowsX).double(); Y3=torch.tensor(rowsY,dtype=torch.float64); F3=torch.tensor(rowsF)
X3b=torch.cat([X3,torch.ones(len(X3),1,dtype=torch.float64)],1)
for lam in [1e-3,1e-2,1e-1,1]:
    A=X3b[F3].T@X3b[F3]+lam*torch.eye(193,dtype=torch.float64); Rr=torch.linalg.solve(A,X3b[F3].T@Y3[F3]); Pv=X3b@Rr; v=~F3
    print('drug-only lam %g val R2 I %.3f N %.3f'%((lam,)+tuple(1-float(((Pv[v,j]-Y3[v,j])**2).mean()/Y3[v,j].var()) for j in (0,1))))
Xall=torch.cat([d['n3'] for d in D]).double(); lab=torch.tensor([1.0 if isdrug(d) else -1.0 for d in D for _ in range(4)],dtype=torch.float64)
Fa=torch.tensor([d['sid'] in FIT for d in D for _ in range(4)])
Xab=torch.cat([Xall,torch.ones(len(Xall),1,dtype=torch.float64)],1)
w=torch.linalg.solve(Xab[Fa].T@Xab[Fa]+1e-2*torch.eye(193,dtype=torch.float64),Xab[Fa].T@lab[Fa]); sc=Xab@w
print('discrim val: drug min/q01 %.2f %.2f | non max/q99 %.2f %.2f'%(float(sc[(~Fa)&(lab>0)].min()),float(sc[(~Fa)&(lab>0)].quantile(.01)),float(sc[(~Fa)&(lab<0)].max()),float(sc[(~Fa)&(lab<0)].quantile(.99))))
pa=Xall@W64['core.prelude.0.ffn.0.weight'].T+W64['core.prelude.0.ffn.0.bias']; ga=Fg(pa)
mx=ga.abs().max(0).values; contrib=ga.abs().mean(0)*W64['core.prelude.0.ffn.2.weight'].norm(dim=0)
print('units max|gelu|<1e-3: %d, <1e-2: %d; smallest contrib q:'%(int((mx<1e-3).sum()),int((mx<1e-2).sum())),np.round(np.sort(contrib.numpy())[:8],5),'median %.3f'%float(contrib.median()))
```