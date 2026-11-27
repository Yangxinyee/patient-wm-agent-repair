
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
