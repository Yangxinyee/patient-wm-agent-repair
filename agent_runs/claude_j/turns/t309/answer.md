Head 2 of the prelude attends to the action token with weight 1.00 at every drug bundle, in all four slots and with no spread.

That makes it a perfect input port. A rank-1 addition to head 2's value projection, (p ⊗ w) with w the rate probe, adds exactly p·(w·LN1(u)) to every slot. It picks up no leakage from other tokens on drug bundles.

Remaining checks before building it:
1. What head 2 attends to on bundles without a drug, where u = 0 and LN1(u) equals the LayerNorm bias β. If it attends to u there too, constraining w·β = 0 makes the injection exactly zero.
2. What `run_cf` returns, and which head-replica functions in heads.py give the query q at readings, so I can track the injected signal through the core to the value head.

```python
c=0; a2=[]
for s in stays[:30]:
    X=T64(X_of(s)); B=BUN[s]
    for k in range(1,len(B)):
        mn,evs=B[k]
        if any(a[0] in ('insulin','norepinephrine') for a in evs): continue
        dt=(B[k][0]-B[k-1][0])*60.0
        hp=X[k-1,192:960].reshape(4,192)*torch.exp(-dt*torch.exp(fa))
        e,u=bvd(evs); z=X[k,960:992]; zt=T64(linp(z,'z_in'))
        tok=torch.cat([hp+SL, T64(e)[None], T64(u)[None], zt.reshape(1,192)],0)
        x=ln1(tok); q=(x[:4]@Wq.T).reshape(4,4,48); kk=(x@Wkv.T)[:,:192].reshape(7,4,48)
        a=torch.softmax(torch.einsum('ihd,jhd->hij',q,kk)/np.sqrt(48),-1); a2.append(a[2].mean(0).numpy())
a2=np.stack(a2); print('nondrug head2 tok-mean', np.round(a2.mean(0),3), 'u-norm', round(float(T64(u).norm()),3))
src=open('/work/heads.py').read(); import re
print(re.findall(r'def (\w+\([^)]*\))', src)[:12])
r=run_cf(stays[0],{},BUN[stays[0]][1][0],X_of(stays[0]))
print(type(r), len(r), [type(x).__name__+str(getattr(x,'shape','')) for x in r][:8])
```