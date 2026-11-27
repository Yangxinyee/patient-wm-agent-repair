`prelude_att(B, rows, EU, k, ret_tokens=True)` returns four things:
- attention (4 heads × 4 slot queries × 7 tokens),
- head outputs (4×4×48),
- raw tokens (7×192),
- n1 tokens (7×192).

That is enough to rebuild the slot stream at the prelude FFN input (x_att = slot tokens + attention output).

Next I collect natural statistics over many bundles: the norm3 outputs of slot tokens, which FFN units are dead on slot tokens, and E. These pick the injection directions and the dead units for the gated decoder. I time the calls first and stop collecting at a time limit.

```python
import time
Wo_t=torch.tensor(Wo); g3t=load('core.prelude.0.norm3.weight').double(); b3t=load('core.prelude.0.norm3.bias').double()
F1w=load('core.prelude.0.ffn.0.weight').double(); F1b=load('core.prelude.0.ffn.0.bias').double()
F2w=load('core.prelude.0.ffn.2.weight').double(); F2b=load('core.prelude.0.ffn.2.bias').double()
gpt=load('core.prelude_norm.weight').double(); bpt=load('core.prelude_norm.bias').double()
def LNd(x,g,b): m=x.mean(-1,keepdim=True); v=((x-m)**2).mean(-1,keepdim=True); return (x-m)/torch.sqrt(v+1e-5)*g+b
t0=time.time(); XA=[];N3=[];EE=[];PRE=[];META=[]
for s in list(dict.fromkeys(SST.tolist()))[:60]:
    Bq,rq,Eq=stay_data(int(s))
    for k in range(len(Bq)):
        att,HO_,tok,n1t=prelude_att(Bq,rq,Eq,k,ret_tokens=True)
        xa=tok[:4].double()+torch.cat([HO_[h] for h in range(4)],-1).double()@Wo_t.T
        n3=LNd(xa,g3t,b3t); pre=n3@F1w.T+F1b; xf=xa+Fnn.gelu(pre)@F2w.T+F2b
        XA.append(xa.numpy()); N3.append(n3.numpy()); EE.append(LNd(xf,gpt,bpt).numpy()); PRE.append(pre.max(0).values.numpy()); META.append((s,k,float(att[2,:,5].mean())))
    if time.time()-t0>400: break
print('bundles',len(XA),'time',round(time.time()-t0,1),'stays',len(set(m[0] for m in META)))
PREm=np.max(np.array(PRE),0); dead=np.where(PREm<-3)[0]; print('dead units (max pre<-3):',len(dead),'max pre of dead', np.round(np.sort(PREm[dead])[-5:],2) if len(dead) else None)
```