The leftover error is mostly in measurement bundles: median 0.041 there against 0.018 on action bundles. Masking absent tokens or storing only measurement bundles in memory doesn't help, so those are not the cause.

Measurement bundles arrive about 55 minutes after the previous bundle, while action bundles arrive 5 minutes after one. So the error may come from a time-dependent part (the time-flow decay of h, or how the decayed h enters the update) rather than from the encoders. I will test variants of where and how the decay is applied, and check whether error grows with the time gap.

- **Confirms:** a decay variant lowers the measurement-bundle error toward the action-bundle level.
- **Refutes:** no variant helps. Then the measurement-value encoding in e is the likelier culprit.

```python
def core_step_v(hprev,dt,e,u,zk,Mem,Mage,dec_mode='std'):
    fa=torch.exp(Wt['core.flow.a'])
    if dec_mode=='std': decay=torch.exp(-dt*fa)
    elif dec_mode=='none': decay=torch.ones(192,dtype=torch.float64)
    elif dec_mode=='hours': decay=torch.exp(-(dt/3600)*fa)
    elif dec_mode=='min': decay=torch.exp(-(dt/60)*fa)
    h=hprev*decay; hu = hprev if dec_mode=='std_upd_raw' else h
    X=torch.cat([h+Wt['core.slots'], e[None], u[None], (zk@Wt['z_in.weight'].T+Wt['z_in.bias'])[None]],0)
    Xn=LNt(X,'core.prelude.0.norm1'); a_,_=mha(Xn,Xn,Xn,'core.prelude.0.attn'); X=X+a_
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
    return y, LNt(y.mean(0),'core.out_norm'), E
print('timescales s: min %.0f max %.3g'%(float(torch.exp(-Wt['core.flow.a']).min()), float(torch.exp(-Wt['core.flow.a']).max())))
dts=np.array([(B0[k][0]-B0[k-1][0])*3600 for k in range(1,len(B0))])
for dm in ['std','none','hours','min']:
    er=[]; erh=[]
    for k in range(1,len(B0)):
        r_=rows0[k]; hp=torch.tensor(SX[rows0[k-1],192:960].reshape(4,192),dtype=torch.float64)
        zk=torch.tensor(SX[r_,960:],dtype=torch.float64); idx_=list(range(k+1))[-128:]
        Mem=torch.stack([EU0[j][0] for j in idx_]); Mage=torch.tensor([(B0[k][0]-B0[j][0])*3600 for j in idx_],dtype=torch.float64)
        y,o,_=core_step_v(hp,dts[k-1],EU0[k][0],EU0[k][1],zk,Mem,Mage,dm)
        er.append(float((o-torch.tensor(SX[r_,:192],dtype=torch.float64)).abs().median()))
        erh.append(float((y-torch.tensor(SX[r_,192:960].reshape(4,192),dtype=torch.float64)).abs().median()))
    er=np.array(er); erh=np.array(erh)
    print(dm,' out med %.4f | dt<600: %.4f  dt>=600: %.4f | h med %.4f'%(np.median(er),np.median(er[dts<600]),np.median(er[dts>=600]),np.median(erh)))
```