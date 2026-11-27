The decoder holds up across the dose range. At ×1, R² is 0.9985 with slope 1.005. At ×½ it is 0.995, and at ×2 it is 0.990 with slope 0.973 and a few saturating outliers. It reads about 0 at norepinephrine-only bundles. Glucose_finger bin edges do not include 150 (one bin spans 143–154), so the per-bin compensation will be approximate within a few bins.

Next I build only the write path for insulin, in a persistent file /work/lib4.py:
- **State side:** cut dimensions 4, 2 and 67.
- **u-encoder:** a rate channel scaled small (0.1 per unit/h) so it does not disturb the other u dimensions.
- **Attention:** head-3 value dimension 20 reads (norm1[4] − norm1[2]), and its output goes only to staging dimension 4.
- **Memory update:** `prelude_norm` and `b_in` feed it into memory dimension 67, which keeps its value at each bundle and decays by time (half-life 1 h).

There is no readout yet, so the factual NLL change measures pure infrastructure cost. What would confirm the write path:
- the per-bundle memory increment is proportional to the start rate (R² >0.98);
- the increment is exactly 0 at non-drug bundles;
- the NLL cost is ≲0.002.

```python
src=r'''
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
'''
open('/work/lib4.py','w').write(src); exec(src,G)
dvec,d0=bd_[:-1],float(bd_[-1])
P1=build_ins(PO,dvec,d0,keep)
ds1=reenc(dsE,P1)
print('u_jU vs 0.1*rstart: max err %.4f | e_jU max %.2e'%(float((ds1['u'][:,jU].numpy()-0.1*rsI)[drug].__abs__().max()), float(ds1['e'][:,[jU,jR]].abs().max())))
H1,O1,st1=tf_loop(P1,ds1,keep=('hf',))
wr=(H1[:,:,jI]-st1['hf'][:,:,jI]).numpy()
cl3=clean3&(rsI>0); sl=np.polyfit(rsI[cl3],wr[cl3].mean(1),1); res=wr[cl3].mean(1)-np.polyval(sl,rsI[cl3])
print('write slope %.3f int %.3f R2 %.4f slot-cv %.3f | nodrug max|w| %.2e | NEonly p99 %.3f'%(sl[0],sl[1],1-res.var()/wr[cl3].mean(1).var(), (wr[cl3].std(1)/wr[cl3].mean(1)).mean(), np.abs(wr[~drug]).max(), np.percentile(np.abs(wr[ne&~ins]),99)))
n0=tf_eval(PO,dsE)[0]; n1_=tf_eval(P1,ds1)[0]
print('NLL orig %.5f edit %.5f diff %.5f (n=%d)'%(n0.mean(),n1_.mean(),n1_.mean()-n0.mean(),len(n0)))
```