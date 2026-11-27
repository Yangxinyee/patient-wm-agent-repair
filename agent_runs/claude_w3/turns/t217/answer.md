The states file has 644 bundle rows. Each row holds out (192), h (4×192) and the z-posterior mean (32), plus stay, hour and drug flags. I also still have my bundle builder and encoder. Next I will recompute the z-posterior locally for every bundle. I take ctx from the previous row's state, then apply `z_post` to [ctx, e, u]. I report the error separately for bundles with and without drug atoms. A small error everywhere would validate the local encoders; a large error only on drug bundles would point at the action encoder.

```python
import torch, math
def gelu(x): return 0.5*x*(1+torch.erf(x/math.sqrt(2)))
def lnf(x,w,b,eps=1e-5):
    m=x.mean(-1,keepdim=True); v=x.var(-1,unbiased=False,keepdim=True); return (x-m)/torch.sqrt(v+eps)*w+b
Wt={n:load(n) for n in NAMES}
def mlp(x,p,act=gelu): return act(x@Wt[p+'.0.weight'].T+Wt[p+'.0.bias'])@Wt[p+'.2.weight'].T+Wt[p+'.2.bias']
X4=torch.tensor(S4['x']); sid4=S4['stay_id']; hr4=S4['hour']
fa_=Wt['core.flow.a']
def ctx_of(o,h,dt):
    hf=h.view(4,192)*torch.exp(-dt*torch.exp(fa_))
    return o+lnf(hf.mean(0),Wt['core.out_norm.weight'],Wt['core.out_norm.bias'])+mlp(torch.tensor([math.log(max(dt,1.0))/10.]),'gap')
res=[]; 
for s in np.unique(sid4):
    idx=np.where(sid4==s)[0]; B=stay_bundles(s)
    if len(B)!=len(idx): res.append(('len',s,len(B),len(idx))); continue
    for a in range(1,len(idx)):
        i,ip=idx[a],idx[a-1]; dt=(hr4[i]-hr4[ip])*3600
        e,u,_,_=enc_bundle(B[a]); e=torch.as_tensor(e,dtype=torch.float32); u=torch.as_tensor(u,dtype=torch.float32)
        c=ctx_of(X4[ip,:192],X4[ip,192:960],dt)
        zp=mlp(torch.cat([c,e,u]),'z_post')[:32]; zt=X4[i,960:]
        res.append((int(S4['has_insulin'][i])+2*int(S4['has_norepinephrine'][i]), float((zp-zt).norm()/zt.norm())))
bad=[r for r in res if r[0]=='len']; print('len mismatch',len(bad), bad[:2])
r=np.array([x for x in res if x[0]!='len'],dtype=float)
for t in range(4): print(t, (r[:,0]==t).sum(), np.median(r[r[:,0]==t,1]).round(3) if (r[:,0]==t).any() else None)
```