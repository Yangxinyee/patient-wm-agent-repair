Stage 1 works at drug bundles:
- The carrier tracks the insulin rate with correlation 0.987 (slope 1.02 against the target 1.0).
- It tracks the norepinephrine rate with correlation 0.981 (slope 0.96).
- Its block-1 readability is lower inside head 1's span than the unconstrained best, about 16 and 15 in signal-to-variation per unit rate.

At non-drug bundles the leak is small on average (|ΔT| ≈ 2.5 against 11 at drug bundles) but reaches 90 at a few bundles. The block-1 units will be gated off there, so this only matters through side effects. I will check it in the factual NLL test rather than chase it now.

Next, stage 2 data with stage 1 active. At every bundle and dose variant I will collect the block-1 FFN input and fit:
1. a drug/no-drug score, which needs a clear held-out margin;
2. readouts of insulin and norepinephrine dose minus expected dose at drug bundles, all variants included, which need high held-out R².

I will also rank the block-1 FFN units by their typical contribution, to pick the three to repurpose.

```python
t0=time.time(); NBR=[]
for d in D:
    s,k=d['sid'],d['k']; r=RK[s][k]; at=bundles_B[s][k][1]
    L_=VK if isdrug(d) else [('',1,1)]
    for sfx,fi,fn_ in L_:
        atv=at if sfx=='' else scale(at,*VN[sfx[1:]]); e,u=enc2(atv)
        nb,_=cstep_K(r['out0'],r['h0'],r['dt'],e,u,r['Mv'],r['Ma'])
        tI=(fi*d['ri']-d['cI']) if isdrug(d) else 0.; tN=(fn_*d['rn']-d['cN']) if isdrug(d) else 0.
        NBR.append((nb.float(),isdrug(d),tI,tN,d['sid'] in FIT,sfx))
print('collected %d in %.0fs'%(len(NBR),time.time()-t0))
Xb=torch.cat([x[0] for x in NBR]).double(); Xb1=torch.cat([Xb,torch.ones(len(Xb),1,dtype=torch.float64)],1)
dr=torch.tensor([x[1] for x in NBR for _ in range(4)]); fi_=torch.tensor([x[4] for x in NBR for _ in range(4)])
Ty=torch.tensor([[x[2],x[3]] for x in NBR for _ in range(4)],dtype=torch.float64)
lab=dr.double()*2-1
for lam in [1e-3,1e-1]:
    w_sc=torch.linalg.solve(Xb1[fi_].T@Xb1[fi_]+lam*torch.eye(193,dtype=torch.float64),Xb1[fi_].T@lab[fi_]); sc=Xb1@w_sc
    v=~fi_; print('lam %g discrim val: drug min %.3f q001 %.3f | non max %.3f q999 %.3f'%(lam,float(sc[v&dr].min()),float(sc[v&dr].quantile(.001)),float(sc[v&~dr].max()),float(sc[v&~dr].quantile(.999))))
m_=fi_&dr
for lam in [1e-4,1e-3,1e-2]:
    w_t=torch.linalg.solve(Xb1[m_].T@Xb1[m_]+lam*torch.eye(193,dtype=torch.float64),Xb1[m_].T@Ty[m_]); pt=Xb1@w_t; v=(~fi_)&dr
    print('lam %g tau val R2 I %.3f N %.3f | resid sd I %.3f N %.3f'%((lam,)+tuple(1-float(((pt[v,j]-Ty[v,j])**2).mean()/Ty[v,j].var()) for j in (0,1))+tuple(float((pt[v,j]-Ty[v,j]).std()) for j in (0,1))))
W1b=W64['core.blocks.0.1.ffn.0.weight']; b1b=W64['core.blocks.0.1.ffn.0.bias']; W2b=W64['core.blocks.0.1.ffn.2.weight']
gb=Fg(Xb[::7]@W1b.T+b1b); contr=gb.abs().mean(0)*W2b.norm(dim=0); order=torch.argsort(contr)
print('block1 ffn smallest contrib',np.round(contr[order[:5]].numpy(),4),'units',order[:5].tolist(),'median %.3f'%float(contr.median()))
```