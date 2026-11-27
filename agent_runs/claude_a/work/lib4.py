
import torch, numpy as np, math
def reenc(ds, Pd):
    g=G; old=g['P']; g['P']=Pd
    try: e,u=g['encode_bundles'](ds['at'], ds['bd']['row'].values.astype(np.int64))
    finally: g['P']=old
    d2=dict(ds); d2['e']=torch.as_tensor(np.asarray(e),dtype=torch.float32); d2['u']=torch.as_tensor(np.asarray(u),dtype=torch.float32); return d2
def build_ins(P0, dvec, d0, keep, jU=4, jR=2, jI=67, k3=20, cu=0.1, cv=6., co=6., gE=5., cb=2., half_life_h=1.0):
    P=G['kill'](P0,[jU,jR,jI],mem=[jI,jR]); pre='core.prelude.0'
    for n in ['norm1','norm3']: P[pre+'.'+n+'.weight'][jI]=0.
    for j in (jU,jR): P[pre+'.norm1.weight'][j]=1.; P[pre+'.norm1.bias'][j]=0.
    P[pre+'.attn.q.weight'][:,[jU,jR]]=0.; P[pre+'.attn.kv.weight'][:,[jU,jR]]=0.
    vr=192+144+k3; kv=P[pre+'.attn.kv.weight']; kv[vr,:]=0.; kv[vr,jU]=cv; kv[vr,jR]=-cv
    Wo=P[pre+'.attn.out.weight']; Wo[:,144+k3]=0.; Wo[jU,144+k3]=co
    for j in (jU,jR): P['encode.norm.weight'][j]=0.; P['encode.norm.bias'][j]=0.
    P['z_in.weight'][[jU,jR],:]=0.; P['z_in.bias'][[jU,jR]]=0.
    ga=P['encode_action.norm.weight']; ba=P['encode_action.norm.bias']; W2=P['encode_action.mlp2.2.weight']; b2=P['encode_action.mlp2.2.bias']
    dv=torch.as_tensor(np.asarray(dvec),dtype=torch.float32); kk=torch.as_tensor(np.asarray(keep))
    coef=dv*ga[kk]; S=float(coef.sum())
    W2[jU,:]=(coef[:,None]*W2[kk,:]).sum(0)/S; b2[jU]=float((coef*b2[kk]).sum())/S
    ga[jU]=cu*S; ba[jU]=cu*(float((dv*ba[kk]).sum())+d0); ga[jR]=0.; ba[jR]=0.
    zp=P['z_post.0.weight']; zp[:,384+jU]=0.; zp[:,384+jR]=0.
    pn=P['core.prelude_norm.weight']; pb=P['core.prelude_norm.bias']; pn[jU]=gE; pn[jR]=gE; pb[jU]=0.; pb[jR]=0.
    P['core.b_in.weight'][jI,jU]=cb; P['core.b_in.weight'][jI,jR]=-cb
    P['core.a_in'][jI]=-30.; P['core.c_out.weight'][jI,jI]=1.
    lam=math.log(2)/(half_life_h*3600.); P['core.flow.a'][jI]=math.log(lam)
    return P

def build_ins2(P0, dvec, d0, keep, jU=4, jR=2, jI=67, k3=20, kap=0.1, gc=1.0, cv=6., co=6., gE=5., cb=1., half_life_h=1.0):
    P=build_ins(P0, dvec, d0, keep, jU, jR, jI, k3, 0.1, cv, co, gE, cb, half_life_h)
    Q=G['cl'](P0)
    W2o=Q['encode_action.mlp2.2.weight']; b2o=Q['encode_action.mlp2.2.bias']; gao=Q['encode_action.norm.weight']; bao=Q['encode_action.norm.bias']
    kk=torch.as_tensor(np.asarray(keep)); dv=torch.as_tensor(np.asarray(dvec),dtype=torch.float32)
    coef=dv*gao[kk]; S=float(coef.sum())
    a=torch.full((192,),-S/192.); a[kk]+=coef
    s_w=kap*(a[:,None]*W2o).sum(0); s_b=kap*float((a*b2o).sum())
    m_w=(W2o[jU]+W2o[jR])/2; m_b=float(b2o[jU]+b2o[jR])/2
    W2=P['encode_action.mlp2.2.weight']; b2=P['encode_action.mlp2.2.bias']
    W2[jU]=m_w+s_w; b2[jU]=m_b+s_b; W2[jR]=m_w-s_w; b2[jR]=m_b-s_b
    c0=d0+float((dv*bao[kk]).sum())
    ga=P['encode_action.norm.weight']; ba=P['encode_action.norm.bias']
    ga[jU]=gc; ga[jR]=gc; ba[jU]=kap*gc*c0; ba[jR]=-kap*gc*c0
    return P
