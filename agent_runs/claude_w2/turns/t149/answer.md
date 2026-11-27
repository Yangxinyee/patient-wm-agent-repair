Useful facts from the vocab:
- The predicted mean is Σ_{b=1..16} p(b)·centre_code[b−1], with no renormalisation.
- `last_status` has a zero padding row and one live row.
- All 8 `last_numeric` columns carry weight (norms ~3), so ARCH's 4 named features must be packed into 8 inputs somehow.

Now that the states are exact, I can test head hypotheses cleanly. Any remaining error then comes from the head definition plus the tool's 4-draw z noise. I'll write a parameterised head and save it to /work/head.py. It covers the type_emb row, the gap scaling, the 8-column feature assignment and the status. First I'll search the type index and gap scaling with the simplest feature guess [has, log1p(age)/10, z_last, 0…]. Success would be one setting that clearly beats the rest, with small and unbiased errors.

```python
code_head = r'''
import numpy as np, torch, torch.nn.functional as F, math
def CEN_of(code_ids):
    cm=np.zeros((13,16))
    for nm,arr in V['value_bin_centres'].items(): cm[CODE[nm]]=np.array(arr)
    return torch.as_tensor(cm[np.asarray(code_ids)],dtype=torch.float32)
def head_base(out_p,h_p,dt_s,W,gdiv=10.0):
    hf=h_p*torch.exp(-dt_s[:,None,None]*torch.exp(W['core.flow.a']))
    return out_p+LN(hf.mean(1),W,'core.out_norm')+mlp(torch.log(dt_s)[:,None]/gdiv,W,'gap')
def head_value_logits(base,codes,f8,lastbin,status,W,z,ty=3):
    view=base+F.linear(z,W['z_head.weight'],W['z_head.bias'])
    q=LN(view+W['type_emb.weight'][ty],W,'query_norm')
    wc=(q+W['embed.code.weight'][codes]+F.linear(f8,W['last_numeric.weight'],W['last_numeric.bias'])
        +W['embed.bins.value.weight'][lastbin]+W['embed.bins.rate.weight'][0]+W['last_status.weight'][status])
    return mlp(wc,W,'attr_heads.value_bin')+W['embed_transition.value_bin.weight'][codes*17+lastbin], wc
def head_probs_mc(base,codes,f8,lastbin,status,W,ty=3,nz=0,seed=0):
    pr=mlp(base,W,'z_prior'); zmu,zls=pr[:,:32],pr[:,32:]
    if nz==0:
        lg,_=head_value_logits(base,codes,f8,lastbin,status,W,zmu,ty); return lg.softmax(-1)
    g=torch.Generator().manual_seed(seed); P=0
    for i in range(nz):
        z=zmu+torch.exp(zls)*torch.randn(zmu.shape,generator=g)
        lg,_=head_value_logits(base,codes,f8,lastbin,status,W,z,ty); P=P+lg.softmax(-1)
    return P/nz
'''
open('/work/head.py','w').write(code_head); exec(code_head)
print('row check', S['stay_id'][fa4.row.values[:3]], S['hour'][fa4.row.values[:3]], S['hour'][fa4.prev.values[:3]])
Xo=torch.as_tensor(Sx[:,:192]); Xh=torch.as_tensor(Sx[:,192:960]).reshape(-1,4,192)
r_=fa4.row.values; p_=fa4.prev.values
dt_t=torch.as_tensor(fa4.dt.values,dtype=torch.float32)
cod=torch.as_tensor(fa4.code.values,dtype=torch.long); lbt=torch.as_tensor(fa4.lb.values,dtype=torch.long)
has_=fa4.has.values.astype(np.float32); age_=fa4.age_s.values.astype(np.float32); zl_=fa4.zl.values.astype(np.float32)
CENr=CEN_of(fa4.code.values); tool_pm=fa4.predicted_mean.values
def f8_make(cols):
    fe={'has':has_,'nohas':1-has_,'la_s':np.log1p(age_)/10,'la_h':np.log1p(age_/3600)/10,'zl':zl_,'zero':0*has_,'one':1+0*has_}
    return torch.as_tensor(np.stack([fe[c] for c in cols],1),dtype=torch.float32)
def pm_score(P):
    pm=(P[:,1:]*CENr).sum(1).numpy(); d=pm-tool_pm
    return np.sqrt(np.mean(d**2)), np.mean(d), pm
f8a=f8_make(['has','la_s','zl','zero','zero','zero','zero','zero'])
st0=torch.zeros(len(fa4),dtype=torch.long)
out_=[]
for gdiv in [10.0,1.0]:
    base=head_base(Xo[p_],Xh[p_],dt_t,W,gdiv)
    for ty_ in range(15):
        P=head_probs_mc(base,cod,f8a,lbt,st0,W,ty=ty_,nz=0)
        rm,me,_=pm_score(P); out_.append((round(rm,3),round(me,3),gdiv,ty_))
out_.sort(); print(out_[:6], out_[-2:])
```