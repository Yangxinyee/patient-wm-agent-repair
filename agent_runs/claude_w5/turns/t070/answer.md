The unit row barely matters once the gap term is dropped; the best error is about 0.093, still about 18% of the signal. So z_post's output may not be the saved posterior mean directly.

Next I'll test common ways of combining prior and posterior: the posterior output alone, the prior mean plus the posterior output, and precision-weighted fusion of the two Gaussians. Each will be tried with and without the gap term. A near-zero error for one would reveal the parameterisation.

```python
res_=[]
for gp in [0,1]:
    for mode in ['post','sum','fuse','fuse_sd']:
        E_=[]
        for p_,c_,dt,sidx,bi in caseinfo:
            out_=X0[p_,:192]; hf=X0[p_,192:960].reshape(4,192)*torch.exp(-dt*flow)
            e_=enc(torch.stack([av4(a_,None,None,True) for a_ in bundles_B[sidx][bi][1]]),'encode')
            ctx=out_+lnp(hf.mean(0),'core.out_norm')+(gapx(np.log(dt)/10) if gp else 0)
            po=mlp(torch.cat([ctx,e_,torch.zeros(192)]),'z_post'); pr=mlp(ctx,'z_prior')
            mq,lq=po[:32],po[32:]; mp,lp=pr[:32],pr[32:]
            if mode=='post': zz=mq
            elif mode=='sum': zz=mp+mq
            elif mode=='fuse':
                wq=torch.exp(-2*lq); wp=torch.exp(-2*lp); zz=(mq*wq+mp*wp)/(wq+wp)
            else:
                wq=torch.exp(-lq); wp=torch.exp(-lp); zz=(mq*wq+mp*wp)/(wq+wp)
            E_.append((zz-X0[c_,960:992]).numpy())
        res_.append((round(float(np.sqrt((np.array(E_)**2).mean())),4),gp,mode))
res_.sort(); print(res_)
```