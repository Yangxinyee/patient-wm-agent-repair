The four helpers are simple: `ln_(x, prefix)`, `lin_(x, prefix)`, `mlp_(x, prefix)` (Linear-GELU-Linear) and `att_(query, key/value, prefix)`. Their prefixes match the weight keys.

Next I will:
- write `cstep_h`, a copy of the core step that also returns h_new (`hn`) and the coda input;
- wrap it in a teacher-forced loop over a stay's bundles that keeps the memory of the last 128 encoder outputs with ages in seconds;
- check it against `cstep` and against the stored GPU states `X_`.

A maximum error near 1e-6 means the hooked version is exact and I can use it for the injection design. A larger error means some detail of the order of operations is off.

```python
import torch, math
import torch.nn.functional as Fn
def cstep_h(out,h,dt_s,e,u,Mv,Mage,inj=None):
    hf=h*torch.exp(-dt_s*torch.exp(W64['core.flow.a']))
    ctx=out+ln_(hf.mean(0),'core.out_norm')+mlp_(torch.tensor([math.log(max(dt_s,1e-9))/10],dtype=torch.float64),'gap')
    zz=mlp_(torch.cat([ctx,e,u]),'z_post')[:32]
    Tk=torch.cat([hf+W64['core.slots'],e[None],u[None],lin_(zz,'z_in')[None]])
    n1=ln_(Tk,'core.prelude.0.norm1'); Tk=Tk+att_(n1,n1,'core.prelude.0.attn')
    Tk=Tk+mlp_(ln_(Tk,'core.prelude.0.norm3'),'core.prelude.0.ffn')
    E=ln_(Tk[:4],'core.prelude_norm'); xb=hf+E
    Mem=Mv+mlp_(torch.log1p(Mage)[:,None],'core.age')
    for l in range(2):
        p='core.blocks.0.%d'%l
        a=ln_(xb,p+'.norm1'); xb=xb+att_(a,a,p+'.attn')
        xb=xb+att_(ln_(xb,p+'.norm2'),Mem,p+'.cross')
        xb=xb+mlp_(ln_(xb,p+'.norm3'),p+'.ffn')
    sp=Fn.softplus(W64['core.delta_in']); abar=torch.exp(-sp*torch.exp(W64['core.a_in']))
    hn=abar*hf+sp*lin_(E,'core.b_in')+(xb-(hf+E))
    if inj is not None: hn=hn+inj
    x=hn@W64['core.c_out.weight'].T
    a=ln_(x,'core.coda.0.norm1'); x2=x+att_(a,a,'core.coda.0.attn')
    y=x2+mlp_(ln_(x2,'core.coda.0.norm3'),'core.coda.0.ffn')
    o=ln_(y.mean(0),'core.out_norm')
    return o,y,zz,E,hf,hn
def run_tf(B,kmax=None,step=cstep_h,keep=False):
    out=torch.zeros(192,dtype=torch.float64); h=torch.zeros(4,192,dtype=torch.float64)
    es=[];ts=[];R=[];tp=None
    for k,(t,atoms) in enumerate(B[:kmax]):
        dt=1.0 if tp is None else (t-tp)*3600.0
        e,u=enc2(atoms); es.append(e); ts.append(t)
        Mv=torch.stack(es[-128:]); Ma=torch.tensor([(t-s)*3600.0 for s in ts[-128:]],dtype=torch.float64)
        res=step(out,h,dt,e,u,Mv,Ma)
        if keep: R.append(dict(out0=out,h0=h,dt=dt,e=e,u=u,Mv=Mv,Ma=Ma,res=res))
        else: R.append(res)
        out,h=res[0],res[1]; tp=t
    return R
sid0=list(bundles_B)[0]; B=bundles_B[sid0]
R1=run_tf(B,30); R2=run_tf(B,30,step=cstep)
d=max(float((a[0]-b[0]).abs().max()) for a,b in zip(R1,R2)); print('vs cstep out maxdiff',d)
print('X_',getattr(X_,'shape',None),'SIDS',getattr(SIDS,'shape',None) if hasattr(SIDS,'shape') else len(SIDS))
```