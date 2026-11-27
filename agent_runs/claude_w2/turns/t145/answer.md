The states file layout is confirmed: 992 numbers per bundle = `out` (192) + 4×192 slots of h + 32-dim z posterior mean. The best stored core-step attempt matched the tool's states only to ~3.8 (correlation 0.9989), so a definitional detail is still wrong. Candidates are the memory contents (whether the current bundle's e is included), the age units/encoding, the gap scaling, and whether u is zeroed for bundles without drugs.

I'll write a fresh, saved harness (/work/core_test.py). It takes the tool's exact state before each bundle and my verified e/u encodings, runs one core step, and compares the result with the tool's state after the bundle, across these variants.
- Confirmed: one variant gives errors near float precision. Then I have an exact core and can do local counterfactual rollouts.
- Refuted: all variants stay at ~1–4. The mismatch then lies elsewhere (e.g. the layer structure).

```python
code_core = r'''
import numpy as np, torch, torch.nn.functional as F, math, pandas as pd
def prep_core(trdf, ids, W, Sx, Ssid, Shr):
    A=make_atoms(trdf[trdf.stay_id.isin(ids)])
    B,E,U,hasu,xa=encode_bundles(A,W)
    return A,B,E,U,hasu
def core_eval(B,E,U,hasu,Sx,W,include_cur=False,age_unit=1.0,age_mode='kadd',age_div=1.0,uzero=False,gdiv=10.0,use_statez=True,idx=None):
    out=torch.as_tensor(Sx[:,:192]); h=torch.as_tensor(Sx[:,192:960]).reshape(-1,4,192); zS=torch.as_tensor(Sx[:,960:992])
    sid=B.stay_id.values; t=B.t.values
    cand=np.where((np.r_[False, sid[1:]==sid[:-1]]))[0] if idx is None else idx
    # memory per bundle: previous bundles of same stay (optionally incl current)
    first=pd.Series(np.arange(len(B))).groupby(sid).transform('min').values
    L=int((cand-first[cand]).max())+1+int(include_cur)
    n=len(cand); Mi=np.full((n,L),-1)
    for r,i in enumerate(cand):
        js=np.arange(first[i], i+1 if include_cur else i)[-128:]
        Mi[r,:len(js)]=js
    mask=torch.as_tensor(Mi>=0); Mi_=np.where(Mi>=0,Mi,0)
    M=E[torch.as_tensor(Mi_)]*mask[...,None]
    age=torch.as_tensor(np.where(Mi>=0,(t[cand][:,None]-t[Mi_])*3600.0,0.0),dtype=torch.float32)
    dt=torch.as_tensor((t[cand]-t[cand-1])*3600.0,dtype=torch.float32)
    u=U[cand].clone()
    if uzero: u=u*torch.as_tensor(hasu[cand])[:,None].float()
    hp=h[cand-1]; op=out[cand-1]
    hf=hp*torch.exp(-dt[:,None,None]*torch.exp(W['core.flow.a']))
    ctx=op+LN(hf.mean(1),W,'core.out_norm')+mlp(torch.log(dt)[:,None]/gdiv,W,'gap')
    z=zS[cand] if use_statez else mlp(torch.cat([ctx,E[cand],U[cand]],-1),W,'z_post')[:,:32]
    X=torch.cat([hf+W['core.slots'][None],E[cand][:,None],u[:,None],F.linear(z,W['z_in.weight'],W['z_in.bias'])[:,None]],1)
    X=layer_sa(X,W,'core.prelude.0'); Ep=LN(X[:,:4],W,'core.prelude_norm')
    agee=mlp(torch.log1p(age/age_unit)[...,None]/age_div,W,'core.age')
    x=hf+Ep
    for l in ['core.blocks.0.0','core.blocks.0.1']: x=layer_blk(x,W,l,M,mask,agee,age_mode)
    sp=F.softplus(W['core.delta_in']); abar=torch.exp(-sp*torch.exp(W['core.a_in']))
    hn=abar*hf+sp*F.linear(Ep,W['core.b_in.weight'])+(x-(hf+Ep))
    y=layer_sa(F.linear(hn,W['core.c_out.weight']),W,'core.coda.0')
    on=LN(y.mean(1),W,'core.out_norm')
    eo=(on-out[cand]).abs().mean().item(); eh=(y-h[cand]).abs().mean().item()
    return eo,eh,cand,on,y
'''
open('/work/core_test.py','w').write(code_core)
exec(open('/work/impl.py').read()); exec(code_core)
W=getW(load_original)
Sx=S['x']; 
A_T,B_T,E_T,U_T,hasu_T=prep_core(tr,selT,W,Sx,S['stay_id'],S['hour'])
print(len(B_T), Sx.shape, np.abs(B_T.t.values-S['hour']).max(), (B_T.stay_id.values==S['stay_id']).mean())
# consistency: out == out_norm(mean h)?
hh_=torch.as_tensor(Sx[:,192:960]).reshape(-1,4,192)
print('out vs LN(mean h):', (LN(hh_.mean(1),W,'core.out_norm')-torch.as_tensor(Sx[:,:192])).abs().max().item())
print('flow tau range (s):', (1/torch.exp(W['core.flow.a'])).min().item(), (1/torch.exp(W['core.flow.a'])).max().item())
res_c=[]
for ic in [False,True]:
  for au_ in [1.0,60.0,3600.0]:
    for am in ['kadd','inadd','kin']:
      for uz in [False,True]:
        eo,eh,_,_,_=core_eval(B_T,E_T,U_T,hasu_T,Sx,W,include_cur=ic,age_unit=au_,age_mode=am,uzero=uz)
        res_c.append((round(eo,4),round(eh,4),ic,au_,am,uz))
res_c.sort(); print(res_c[:8]); print(res_c[-3:])
```