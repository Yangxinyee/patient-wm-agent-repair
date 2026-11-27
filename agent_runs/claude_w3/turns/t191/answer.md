Head 2 attends to the action token almost exclusively at drug bundles (p ≈ 0.99, sd 0.035). That makes it a clean injection route. Next I'll write my own z-posterior check, testing both GELU and ReLU, so I can test how END atoms are handled. The large errors were in END-only bundles.

```python
def mlp2l(x,pre,act):
    h=x@T(pre+'.0.weight').T+T(pre+'.0.bias'); h=torch.nn.functional.gelu(h) if act=='gelu' else torch.relu(h)
    return h@T(pre+'.2.weight').T+T(pre+'.2.bias')
def zpost_mean(outp,hprev,dt,e,u,act='gelu'):
    hf=hprev*torch.exp(-dt*torch.exp(fa_))
    on=lnorm(hf.mean(0),T('core.out_norm.weight'),T('core.out_norm.bias'))
    g=mlp2l(torch.tensor([[np.log(dt)/10]]).double(),'gap',act)[0]
    ctx=outp+on+g
    return mlp2l(torch.cat([ctx,e,u])[None],'z_post',act)[0,:32]
def zerr_stays(sids,act='gelu',mod=None,maxk=999):
    res={}
    for sid in sids:
        rows=np.where(S4['stay_id']==sid)[0]; Bs=stay_bundles(int(sid))
        for k in range(1,min(len(rows),maxk)):
            bb=Bs[k]; bb2=mod(bb) if mod else bb
            e,u,_,_=enc_bundle(bb2); dt=(bb['minute'].iloc[0]-Bs[k-1]['minute'].iloc[0])*60
            zm=zpost_mean(torch.tensor(S4['x'][rows[k-1],:192]).double(),torch.tensor(Hs[rows[k-1]]).double(),dt,e.double(),u.double(),act)
            zt=torch.tensor(Zs[rows[k]]).double()
            ty='+'.join(sorted(set(f"{'A' if c==8 else 'B'}{int(kd)}" for c,kd in zip(bb['code'],bb['kind']) if c in (8,9)))) or 'M'
            res.setdefault(ty,[]).append(float((zm-zt).norm()/zt.norm()))
    return {k:(len(v),round(float(np.median(v)),3)) for k,v in res.items()}
sids=np.unique(S4['stay_id'])[:4]
for act in ['gelu','relu']: print(act, zerr_stays(sids,act))
```