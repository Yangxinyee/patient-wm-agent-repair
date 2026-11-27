The best variant reproduces the model's `out` with a median error of 0.028 (worst bundle 0.07), against element sizes of order 1. That variant measures memory age in seconds as log(1+age) and adds the age term to the memory vectors before both the key and value projections, with no memory normalisation. The current bundle's e is included in memory.

That is close but not exact, so something is still slightly off. Candidates are tokens for an absent e or u, memory entries for bundles without measurements, or my action encoding.

Next I will split the per-bundle error by bundle type, insulin/norepi vs not, and test two masking variants: dropping absent e/u tokens from the prelude, and storing only bundles that have measurements in memory.

- **Confirms an encoding problem:** errors concentrate on action bundles, which would point at u_k, as the old z mismatch did.
- **Points elsewhere:** errors are uniform. Then the issue is a structural detail.

```python
def run_stay(B,rows,EU,var,mask_prelude=False,mem_obs_only=False,ret=False):
    errs=[]; outs=[]
    for k in range(1,len(B)):
        r_=rows[k]; hp=torch.tensor(SX[rows[k-1],192:960].reshape(4,192),dtype=torch.float64); op=torch.tensor(SX[rows[k-1],:192],dtype=torch.float64)
        dt=(B[k][0]-B[k-1][0])*3600; zk=torch.tensor(SX[r_,960:],dtype=torch.float64)
        hasobs=[any(not (a['ek']==2 and a['kind']==0) for a in B[j][1]) for j in range(k+1)]
        idx_=[j for j in range(k+1) if (hasobs[j] or not mem_obs_only)][-128:]
        Mem=torch.stack([EU[j][0] for j in idx_]) if idx_ else None
        Mage=torch.tensor([(B[k][0]-B[j][0])*3600 for j in idx_],dtype=torch.float64) if idx_ else None
        e_,u_=EU[k]
        if mask_prelude:
            y,o,_=core_step_m(hp,op,dt,e_,u_,zk,Mem,Mage,var, use_e=hasobs[k], use_u=any(a['ek']==2 and a['kind']==0 for a in B[k][1]))
        else:
            y,o,_=core_step(hp,op,dt,e_,u_,zk,Mem,Mage,var)
        errs.append(float((o-torch.tensor(SX[r_,:192],dtype=torch.float64)).abs().median()))
    return np.array(errs)
import inspect
src_cs=inspect.getsource(core_step) if False else None
def core_step_m(hprev,outprev,dt,e,u,zk,Mem,Mage,var,use_e=True,use_u=True):
    decay=torch.exp(-dt*torch.exp(Wt['core.flow.a'])); h=hprev*decay
    toks=[h+Wt['core.slots']]+([e[None]] if use_e else [])+([u[None]] if use_u else [])+[(zk@Wt['z_in.weight'].T+Wt['z_in.bias'])[None]]
    X=torch.cat(toks,0)
    Xn=LNt(X,'core.prelude.0.norm1'); a_,_=mha(Xn,Xn,Xn,'core.prelude.0.attn'); X=X+a_
    X=X+ffn(LNt(X,'core.prelude.0.norm3'),'core.prelude.0.ffn'); E=LNt(X[:4],'core.prelude_norm')
    x=h+E; AE=age_emb(Mage,var['sc']) if Mem is not None else None
    for L in ['core.blocks.0.0','core.blocks.0.1']:
        xn=LNt(x,L+'.norm1'); a_,_=mha(xn,xn,xn,L+'.attn'); x=x+a_
        if Mem is not None:
            xn=LNt(x,L+'.norm2'); a_,_=mha(xn,Mem+AE,Mem+AE,L+'.cross'); x=x+a_
        x=x+ffn(LNt(x,L+'.norm3'),L+'.ffn')
    sp=F.softplus(Wt['core.delta_in']); abar=torch.exp(-sp*torch.exp(Wt['core.a_in']))
    hn=abar*h+sp*(E@Wt['core.b_in.weight'].T)+(x-(h+E))
    y=hn@Wt['core.c_out.weight'].T
    yn=LNt(y,'core.coda.0.norm1'); a_,_=mha(yn,yn,yn,'core.coda.0.attn'); y=y+a_
    y=y+ffn(LNt(y,'core.coda.0.norm3'),'core.coda.0.ffn')
    return y, LNt(y.mean(0),'core.out_norm'), E
best=dict(sc=scs['s'],kf='both',mnorm=False)
isact=np.array([any(a['ek']==2 and a['kind']==0 for a in B0[k][1]) for k in range(1,len(B0))])
for mp,mo in itertools.product([False,True],[False,True]):
    er=run_stay(B0,rows0,EU0,best,mask_prelude=mp,mem_obs_only=mo)
    print('mask_prelude',mp,'mem_obs_only',mo,' med all %.4f  act %.4f  nonact %.4f  max %.4f'%(np.median(er),np.median(er[isact]),np.median(er[~isact]),er.max()))
```