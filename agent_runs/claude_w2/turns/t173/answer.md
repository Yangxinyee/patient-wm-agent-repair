`tr` is the training frame in the format `prep_rollout` expects. `Kins` and `Knor` are my 12-lag structural impulse responses: insulin on glucose -2.0, -2.5, -2.2, … per U/h·hour, and norepinephrine on MAP +3.6, +2.7, +1.7, ….

My plan is a correction added to the value head. It should be about 0 on factual histories and equal to the structural one-step response minus the model's own response when doses differ from what the readings imply. The correction has to be read linearly from the head input, so the open question is decodability. To test it I run counterfactual teacher forcing: recorded readings with doses scaled from various hours, plus random per-infusion multipliers, on about 300 training stays. I store every head input.

If a linear decoder of the head input captures the dose changes well (high interventional R²), the approach is viable. If not, I need to build a dedicated state channel instead. Next I write the collection code to a file and run it.

```python
code_cf = r'''
import numpy as np, torch, pandas as pd, time
def arm_embed(R,W,arm):
    A=R['A'].copy()
    for drug,key in [('insulin','ins'),('norepinephrine','ne')]:
        m=((A.code==CODE[drug])&(A.kind==0)).values
        if key+'_mult' in arm:
            mult=np.asarray(arm[key+'_mult'],dtype=float)
            A.loc[m,'rate']=A.rate.values[m]*mult[m]; A.loc[m,'dose']=A.dose.values[m]*mult[m]; continue
        f,h0=arm.get(key,1.0),arm.get(key+'_from',12.0)
        m=m&(A.t.values>=h0)
        if arm.get(key+'_drop',False) and f==0: A.loc[m,['rate','dose']]=np.nan
        else: A.loc[m,'rate']=A.rate.values[m]*f; A.loc[m,'dose']=A.dose.values[m]*f
    return embed(A,atom_features(A),W)
def arm_mult(R,arm,drug):
    """per-row multiplier actually applied to drug START rows"""
    A=R['A']; key={'insulin':'ins','norepinephrine':'ne'}[drug]
    m=((A.code==CODE[drug])&(A.kind==0)).values; mult=np.ones(len(A))
    if key+'_mult' in arm: mult=np.where(m,np.asarray(arm[key+'_mult'],dtype=float),1.0); return mult
    f,h0=arm.get(key,1.0),arm.get(key+'_from',12.0); mult[m&(A.t.values>=h0)]=f; return mult
def dose_grid(R,drug,mult=None):
    """G[si, h+3] = rate of infusion starting in hour h (h=-3..27)"""
    A=R['A']; m=((A.code==CODE[drug])&(A.kind==0)).values
    G=np.zeros((len(R['stays']),32))
    r=A.rate.values[m]*(1.0 if mult is None else mult[m]); hr=np.floor(A.t.values[m]+1e-6).astype(int)
    np.add.at(G,(A.si.values[m],hr+3),np.nan_to_num(r))
    return G
def cf_collect(df,ids,arms,W,seed=0):
    R=prep_rollout(df,ids); store=[]; t0=time.time()
    res=run_rollout(R,W,arms,n_paths=1,seed=seed,t_hide=1e9,want_nll=True,store=store)
    HN=np.concatenate([s[0] for s in store]); HJ=np.concatenate([s[1] for s in store])
    D=dict(n=HN,j=HJ,base=torch.cat([s[2] for s in store]),P=torch.cat([s[3] for s in store]),
           f8=torch.cat([s[4] for s in store]),lb=torch.cat([s[5] for s in store]),zmu=torch.cat([s[6] for s in store]))
    nA=len(arms); D['si']=HN//nA; D['arm']=HN%nA
    A=R['A']; D['code']=A.code.values[HJ]; D['t']=A.t.values[HJ]; D['value']=A.value.values[HJ]
    cen=torch.as_tensor(CENT[D['code']],dtype=torch.float32); D['pm']=(D['P'][:,1:]*cen).sum(1).numpy()
    print('collect %.1fs, %d head rows'%(time.time()-t0,len(HN)))
    return R,res,D
'''
open('/work/cf.py','w').write(code_cf); exec(code_cf)
# choose 300 train stays with insulin or NE from hour>=6
dr_=tr[(tr.event.isin(['insulin','norepinephrine']))&(tr.hour>=6)]
cand_ids=np.array(sorted(dr_.stay_id.unique()))
rng_=np.random.default_rng(7); idsC=list(rng_.choice(cand_ids,300,replace=False))
R0=prep_rollout(tr,idsC[:1]); print(R0['A'].columns.tolist())
Rtmp=prep_rollout(tr,idsC)
nrow=len(Rtmp['A']); r1=np.exp(rng_.uniform(np.log(0.25),np.log(3.0),nrow)); r2=np.exp(rng_.uniform(np.log(0.25),np.log(3.0),nrow))
armsC=[{}, {'ins':2.0},{'ins':0.5},{'ins':0.0},{'ins':3.0},{'ins':2.0,'ins_from':6.0},{'ins':0.5,'ins_from':16.0},
       {'ne':2.0},{'ne':0.5},{'ne':0.0},{'ne':3.0},{'ne':2.0,'ne_from':6.0},{'ins_mult':r1},{'ne_mult':r2}]
RC,resC,DC=cf_collect(tr,idsC,armsC,W)
print({k:(v.shape if hasattr(v,'shape') else len(v)) for k,v in DC.items()})
```