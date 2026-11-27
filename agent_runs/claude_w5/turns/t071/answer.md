Fusing prior and posterior by precision helps only slightly (0.086 against 0.094), and adding the gap term makes every version worse. The exact parameterisation is still unknown.

Next I'll check whether the two z networks use a different activation from the rest. I'll try four activations on z_post and z_prior, crossed with posterior-only versus fused, with no gap term. A sharp drop toward zero would identify it; if nothing changes, the leftover error comes from elsewhere.

```python
def mlpX(x,p,af): return lin(af(lin(x,p+'.0')),p+'.2')
res_=[]
pre=[]
for p_,c_,dt,sidx,bi in caseinfo:
    out_=X0[p_,:192]; hf=X0[p_,192:960].reshape(4,192)*torch.exp(-dt*flow)
    e_=enc(torch.stack([av4(a_,None,None,True) for a_ in bundles_B[sidx][bi][1]]),'encode')
    pre.append((out_+lnp(hf.mean(0),'core.out_norm'),e_,X0[c_,960:992]))
for an,af in acts.items():
    for mode in ['post','fuse']:
        E_=[]
        for ctx,e_,zt_ in pre:
            po=mlpX(torch.cat([ctx,e_,torch.zeros(192)]),'z_post',af); pr=mlpX(ctx,'z_prior',af)
            zz=po[:32] if mode=='post' else (po[:32]*torch.exp(-2*po[32:])+pr[:32]*torch.exp(-2*pr[32:]))/(torch.exp(-2*po[32:])+torch.exp(-2*pr[32:]))
            E_.append((zz-zt_).numpy())
        res_.append((round(float(np.sqrt((np.array(E_)**2).mean())),4),an,mode))
res_.sort(); print(res_)
```