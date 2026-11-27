Both unbounded write locations have very anisotropic natural variation:
- **Block-1 FFN input:** variance eigenvalues run from 18.5 down to a median of 7.5e-5.
- **Coda FFN input:** from 5.5 down to a median of 4e-6.

So there are quiet directions. A unit change in E can be read there with a signal-to-variation ratio of about 26 (block-1 input) or 17 (coda input).

That makes a two-hop design workable:
1. The prelude places a dose signal in a carrier direction of E.
2. A few GELU-gated units in the last block's FFN read the carrier, the drug/no-drug score and the expected dose from their own input.
3. Those units write the belief shift straight into h_new, with no size limit.

The carrier must be large enough to read cleanly (|ΔE| of about 0.5 per unit rate) but small enough not to distort the prelude's LayerNorm.

Next I will compute, at 15 drug bundles, the composite Jacobian from a slot-residual write in the prelude (post-attention, pre-FFN) to the block-1 FFN input. Then I solve for the best signal-to-variation ratio per unit write, using the natural covariance at drug bundles. I also print the spread of the prelude LayerNorm gains.

A high ratio per unit |ΔT| would let the write stay a small fraction of |T| ≈ 760.

```python
def cstep_K(out,h,dt_s,e,u,Mv,Mage,dT=None):
    hf=h*torch.exp(-dt_s*torch.exp(W64['core.flow.a']))
    ctx=out+ln_(hf.mean(0),'core.out_norm')+mlp_(torch.tensor([math.log(max(dt_s,1e-9))/10],dtype=torch.float64),'gap')
    zz=mlp_(torch.cat([ctx,e,u]),'z_post')[:32]
    Tk=torch.cat([hf+W64['core.slots'],e[None],u[None],lin_(zz,'z_in')[None]])
    n1=ln_(Tk,'core.prelude.0.norm1'); Tk=Tk+att_(n1,n1,'core.prelude.0.attn')
    if dT is not None: Tk=torch.cat([Tk[:4]+dT,Tk[4:]])
    Tk=Tk+mlp_(ln_(Tk,'core.prelude.0.norm3'),'core.prelude.0.ffn')
    E=ln_(Tk[:4],'core.prelude_norm'); xb=hf+E; Mem=Mv+mlp_(torch.log1p(Mage)[:,None],'core.age')
    for l in range(2):
        p='core.blocks.0.%d'%l
        a=ln_(xb,p+'.norm1'); xb=xb+att_(a,a,p+'.attn'); xb=xb+att_(ln_(xb,p+'.norm2'),Mem,p+'.cross')
        nb=ln_(xb,p+'.norm3'); xb=xb+mlp_(nb,p+'.ffn')
    return nb,E
gp=W64['core.prelude_norm.weight']; print('prelude_norm gamma |.| q10/50/90',np.round(np.quantile(gp.abs().numpy(),[.1,.5,.9]),3))
dsel=[d for d in D if isdrug(d) and d['sid'] in FIT][::80][:15]; KB=[];KE=[]
for d in dsel:
    s,k=d['sid'],d['k']; r=RK[s][k]; e,u=enc2(bundles_B[s][k][1]); nb0,E0=cstep_K(r['out0'],r['h0'],r['dt'],e,u,r['Mv'],r['Ma'])
    kb=torch.zeros(4,192,192,dtype=torch.float64); ke=torch.zeros_like(kb)
    for j in range(192):
        dT_=torch.zeros(4,192,dtype=torch.float64); dT_[:,j]=1.0
        nb1,E1=cstep_K(r['out0'],r['h0'],r['dt'],e,u,r['Mv'],r['Ma'],dT=dT_); kb[:,:,j]=nb1-nb0; ke[:,:,j]=E1-E0
    KB.append(kb); KE.append(ke)
NBd=torch.cat([NB[4*i:4*i+4] for i,d in enumerate(D[::3]) if isdrug(d)])
Sb=torch.cov(NBd.T)+1e-9*torch.eye(192,dtype=torch.float64)
Km=torch.stack(KB).reshape(-1,192,192).mean(0); Kem=torch.stack(KE).reshape(-1,192,192).mean(0)
Mk=Km.T@torch.linalg.solve(Sb,Km); evk,veck=torch.linalg.eigh(Mk)
print('SNR^2 per unit |dT| top3',np.round(evk[-3:].numpy(),2),'| |dE|/|dT| along top dir %.4f'%float((Kem@veck[:,-1]).norm()))
```