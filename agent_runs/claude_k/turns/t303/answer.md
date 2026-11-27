Pass 12 gave the original and doubled-dose states for the same 99 stays, with identical readings, so any difference in q at a reading is the model's direct response to the doses.

Plan for this analysis:
1. Rebuild q at every reading from both state sets, and check the original against my stored q (Q99) to confirm the rows align.
2. Attach the recorded rates of the latest and second-latest insulin and norepinephrine bundles before each reading.
3. Regress Δq on those rates, separately at glucose readings and at MAP readings, to get the per-unit response direction.
4. Compute the GLS readout direction w = Σ⁻¹a against the natural covariance of the value-head input, and its signal-to-noise ratio per unit².

The readout route is worthwhile if the noise of the readout, 1/√SNR in dose units, is small next to typical dose changes (insulin about 1.7 units/h, norepinephrine about 0.5). It is hopeless if the noise is several times larger.

```python
import numpy as np, pandas as pd, json
A=np.load('/work/fwd_12/states_original.npz'); E_=np.load('/work/fwd_12/states_edited.npz')
xo=A['x'].astype(np.float64); xe=E_['x'].astype(np.float64)
print('R cols', R.columns.tolist(), len(R))
vf=view_from(xo[int(R.k.iloc[0])], float(R.dt.iloc[0])); print('view_from ->', type(vf).__name__, [np.shape(v) for v in vf] if isinstance(vf,(tuple,list)) else np.shape(vf))
Qo=np.array([view_from(xo[int(k)],float(dt))[0] for k,dt in zip(R.k,R.dt)]); Qe=np.array([view_from(xe[int(k)],float(dt))[0] for k,dt in zip(R.k,R.dt)])
print('Qo vs Q99 maxdiff %.2e'%np.abs(Qo-Q99).max()); dQ=Qe-Qo
ids=[int(i) for i in json.load(open('/work/sel_states.json'))]
trd=pd.read_csv('/data/train.csv',usecols=['stay_id','hour','event','rate']); trd=trd[trd.stay_id.isin(ids)&trd.event.isin(['insulin','norepinephrine'])]
sid=A['stay_id']; hr=A['hour'].astype(float); hi=A['has_insulin'].astype(bool); hn=A['has_norepinephrine'].astype(bool)
mp={nm:{(int(s),int(h)):float(r) for s,h,r in trd[trd.event==nm][['stay_id','hour','rate']].itertuples(index=False)} for nm in ['insulin','norepinephrine']}
rateI=np.array([mp['insulin'].get((int(s),int(np.floor(h))),np.nan) if f else np.nan for s,h,f in zip(sid,hr,hi)])
rateN=np.array([mp['norepinephrine'].get((int(s),int(np.floor(h))),np.nan) if f else np.nan for s,h,f in zip(sid,hr,hn)])
print('matched rates: ins %d/%d ne %d/%d'%(np.isfinite(rateI[hi]).sum(),hi.sum(),np.isfinite(rateN[hn]).sum(),hn.sum()))
def lastrows(flag):
    l1=np.full(len(sid),-1); l2=np.full(len(sid),-1); c1=c2=-1; ps=None
    for b in range(len(sid)):
        if sid[b]!=ps: c1=c2=-1; ps=sid[b]
        if flag[b]: c2=c1; c1=b
        l1[b]=c1; l2[b]=c2
    return l1,l2
lI1,lI2=lastrows(hi); lN1,lN2=lastrows(hn)
kk=R.k.values.astype(int)
def feat(l,rate): return np.array([rate[l[k]] if l[k]>=0 and np.isfinite(rate[l[k]]) else 0.0 for k in kk])
F=np.stack([feat(lI1,rateI),feat(lI2,rateI),feat(lN1,rateN),feat(lN2,rateN)],1)
codecol=[c for c in R.columns if c.lower() in ('code','code_id','event','c')]; print('code col', codecol)
cv=R[codecol[0]].values
isG=np.isin(cv,[6,10,'glucose_finger','glucose_lab']); isM=np.isin(cv,[5,7,'map_cuff','map_arterial']); print('n gluc',isG.sum(),'n map',isM.sum())
res={}
for nm,msk,col in [('INS@gluc',isG,0),('NE@map',isM,2)]:
    Fm=F[msk]; Y=dQ[msk]; Bc,_,_,_=np.linalg.lstsq(Fm,Y,rcond=None); fit=Fm@Bc
    print(nm,'dQ rms %.4f fitR2 %.3f |a lag1| %.4f |a lag2| %.4f |other drug lag1| %.4f'%(np.sqrt((Y**2).sum(1)).mean(),1-((Y-fit)**2).sum()/(Y**2).sum(),np.linalg.norm(Bc[col]),np.linalg.norm(Bc[col+1]),np.linalg.norm(Bc[2-col])))
    a=Bc[col]; S=np.cov(WC99[msk].T)+1e-4*np.eye(192); Si_a=np.linalg.solve(S,a); snr=float(a@Si_a); w=Si_a/snr
    pr=dQ[msk]@w; print('   SNR/unit^2 %.3f noise sd (dose units) %.3f | corr(w.dQ, r lag1) %.3f slope %.3f | natural q var tot %.3f'%(snr,1/np.sqrt(snr),np.corrcoef(pr,Fm[:,col])[0,1],np.polyfit(Fm[:,col],pr,1)[0],np.trace(S)))
    res[nm]=dict(a=a,w=w,snr=snr,c0=float((WC99[msk]@w).mean()))
```