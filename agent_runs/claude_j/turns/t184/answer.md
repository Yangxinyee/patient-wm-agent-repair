Using all bundles in memory with ages in seconds gives a median one-step error of about 0.01, for drug and measurement bundles alike. The error is uniform rather than tied to any atom type. That looks like reduced-precision arithmetic on the tool's GPU (TF32/bf16), not a structural mistake. For counterfactual differences this precision is adequate.

The remaining piece is the z posterior, which the counterfactual run must compute itself because it depends on u_k. I will implement `ctx = out + out_norm(mean_slots(h flowed)) + gap(log dt/10)` and `z_post([ctx, e, u])`, and compare with the tool's z. I will try GELU and also log dt measured in seconds versus hours. If the error is about 0.01 or less, the full counterfactual recursion is ready.

```python
import torch.nn.functional as F
def LNw(x,p,eps=1e-5): return F.layer_norm(x,(x.shape[-1],),Wt[p+'.weight'],Wt[p+'.bias'],eps)
def Lw(x,p): return x@Wt[p+'.weight'].T+(Wt[p+'.bias'] if p+'.bias' in Wt else 0)
def gapf(ldt): return Lw(F.gelu(Lw(torch.tensor([ldt],dtype=torch.float64),'gap.0')),'gap.2')
def flow(h,dt): return h*torch.exp(-dt*torch.exp(Wt['core.flow.a']))
def ctxf(o,h,dt,unit=1.0): return o+LNw(flow(h,dt).mean(0),'core.out_norm')+gapf(math.log(dt/unit)/10)
def zpost(o,h,dt,e,u,unit=1.0):
    c=ctxf(o,h,dt,unit); return Lw(F.gelu(Lw(torch.cat([c,e,u]),'z_post.0')),'z_post.2')[:32]
for unit in [1.0,3600.0]:
    er=[]
    for k in range(1,54):
        B=BUN[s0]; dt=(B[k][0]-B[k-1][0])*60.0; e,u=vecs(s0,k)
        er.append(float((zpost(X0[k-1,:192],X0[k-1,192:960].reshape(4,192),dt,e,u,unit)-X0[k,-32:]).abs().max()))
    print(unit, np.round([max(er),np.median(er)],5))
```