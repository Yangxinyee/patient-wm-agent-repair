The z-Jacobian is essentially zero for every dimension. That cannot be right for the real model: z dims 9 and 20 vary strongly (sd about 1.9), and a posterior latent that never changed the state would have collapsed during training.

The likely flaw is in my prelude. The slot tokens carry h with norm about 800, so after the normalisations the small contributions from the e, u and z tokens vanish relative to h. That means my input to the prelude slots is probably wrong (for example h should be normalised first, or not added), and the leftover 3% error on `out` is the missing information from these tokens.

Next I will test prelude slot-input variants and keep whichever cuts the error:
- h + slots (current)
- slots only
- h normalised with out_norm, plus slots
- h normalised without parameters, plus slots

```python
def core_step_p(hprev,dt,e,u,zk,Mem,Mage,pv):
    decay=torch.exp(-dt*torch.exp(Wt['core.flow.a'])); h=hprev*decay
    if pv=='h+slots': S=h+Wt['core.slots']
    elif pv=='slots': S=Wt['core.slots'].clone()
    elif pv=='outnorm(h)+slots': S=LNt(h,'core.out_norm')+Wt['core.slots']
    elif pv=='ln(h)+slots': S=F.layer_norm(h,(192,))+Wt['core.slots']
    X=torch.cat([S, e[None], u[None], (zk@Wt['z_in.weight'].T+Wt['z_in.bias'])[None]],0)
    Xn=LNt(X,'core.prelude.0.norm1'); a_,att=mha(Xn,Xn,Xn,'core.prelude.0.attn'); X=X+a_
    X=X+ffn(LNt(X,'core.prelude.0.norm3'),'core.prelude.0.ffn'); E=LNt(X[:4],'core.prelude_norm')
    x=h+E; AE=age_emb(Mage,scs['s'])
    for L in ['core.blocks.0.0','core.blocks.0.1']:
        xn=LNt(x,L+'.norm1'); a_,_=mha(xn,xn,xn,L+'.attn'); x=x+a_
        xn=LNt(x,L+'.norm2'); a_,_=mha(xn,Mem+AE,Mem+AE,L+'.cross'); x=x+a_
        x=x+ffn(LNt(x,L+'.norm3'),L+'.ffn')
    sp=F.softplus(Wt['core.delta_in']); abar=torch.exp(-sp*torch.exp(Wt['core.a_in']))
    hn=abar*h+sp*(E@Wt['core.b_in.weight'].T)+(x-(h+E))
    y=hn@Wt['core.c_out.weight'].T
    yn=LNt(y,'core.coda.0.norm1'); a_,_=mha(yn,yn,yn,'core.coda.0.attn'); y=y+a_
    y=y+ffn(LNt(y,'core.coda.0.norm3'),'core.coda.0.ffn')
    return y, LNt(y.mean(0),'core.out_norm'), E, att
for pv in ['h+slots','slots','outnorm(h)+slots','ln(h)+slots']:
    eo=[]; eh=[]; attz=[]
    for k in range(1,len(B0)):
        r_=rows0[k]; hp=torch.tensor(SX[rows0[k-1],192:960].reshape(4,192),dtype=torch.float64); zk=torch.tensor(SX[r_,960:],dtype=torch.float64)
        idx_=list(range(k+1))[-128:]; Mem=torch.stack([EU0[i][0] for i in idx_]); Mage=torch.tensor([(B0[k][0]-B0[i][0])*3600 for i in idx_],dtype=torch.float64)
        y,o,E,att=core_step_p(hp,(B0[k][0]-B0[k-1][0])*3600,EU0[k][0],EU0[k][1],zk,Mem,Mage,pv)
        ht=torch.tensor(SX[r_,192:960].reshape(4,192),dtype=torch.float64)
        eo.append(float((o-torch.tensor(SX[r_,:192],dtype=torch.float64)).abs().median())); eh.append(float((y.mean(0)-ht.mean(0)).norm()/ht.mean(0).norm()))
        attz.append(att[:,:4,4:].sum(-1).mean().item())
    print('%-18s out med %.4f  h rel med %.4f  slot->(e,u,z) attn %.3f'%(pv,np.median(eo),np.median(eh),np.mean(attz)))
```