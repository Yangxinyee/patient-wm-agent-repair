Run 281 failed for the same kind of reason as run 275. The kernel's `view_from` calls a weight helper `L`, which relies on a global I overwrote in run 273, when I bound `A` to an npz file. I will:
- inspect what `L` and `mlpf` use and restore `A` as a tensor accessor;
- load the state files under names that cannot collide;
- rerun the analysis.

The original-model check against Q99 will confirm the restored helpers reproduce the verified q exactly. If they did not, the other single-letter globals I overwrote (`P`, `V`, `X`) would be the suspects.

```python
import numpy as np, pandas as pd, json
for fn in ['L', 'mlpf']:
    f = globals().get(fn); print(fn, f.__code__.co_varnames[:8], 'names', f.__code__.co_names, 'consts', f.__code__.co_consts[:8])
def A(n): return T[n]
SO=np.load('/work/fwd_12/states_original.npz'); SE=np.load('/work/fwd_12/states_edited.npz')
xo=SO['x'].astype(np.float64); xe=SE['x'].astype(np.float64)
Qo=np.array([view_from(xo[int(k)],float(dt))[0] for k,dt in zip(R.k,R.dt)]); print('Qo vs Q99 maxdiff %.2e'%np.abs(Qo-Q99).max())
Qe=np.array([view_from(xe[int(k)],float(dt))[0] for k,dt in zip(R.k,R.dt)]); dQ=Qe-Qo
ids=[int(i) for i in json.load(open('/work/sel_states.json'))]
trd=pd.read_csv('/data/train.csv',usecols=['stay_id','hour','event','rate']); trd=trd[trd.stay_id.isin(ids)&trd.event.isin(['insulin','norepinephrine'])]
sid=SO['stay_id']; hr=SO['hour'].astype(float); hi=SO['has_insulin'].astype(bool); hn=SO['has_norepinephrine'].astype(bool)
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
lI1,lI2=lastrows(hi); lN1,lN2=lastrows(hn); kk=R.k.values.astype(int)
def feat(l,rate): return np.array([rate[l[k]] if l[k]>=0 and np.isfinite(rate[l[k]]) else 0.0 for k in kk])
FD=np.stack([feat(lI1,rateI),feat(lI2,rateI),feat(lN1,rateN),feat(lN2,rateN)],1)
ev=R['event'].values; isG=np.isin(ev,['glucose_finger','glucose_lab']); isM=np.isin(ev,['map_arterial','map_cuff']); print('n gluc',isG.sum(),'n map',isM.sum())
res={}
for nm,msk,col in [('INS@gluc',isG,0),('NE@map',isM,2)]:
    Fm=FD[msk]; Y=dQ[msk]; Bc=np.linalg.lstsq(Fm,Y,rcond=None)[0]; fit=Fm@Bc
    print(nm,'dQ rms %.4f fitR2 %.3f |a1| %.4f |a2| %.4f |other1| %.4f'%(np.sqrt((Y**2).sum(1)).mean(),1-((Y-fit)**2).sum()/(Y**2).sum(),np.linalg.norm(Bc[col]),np.linalg.norm(Bc[col+1]),np.linalg.norm(Bc[2-col])))
    a=Bc[col]; Sg=np.cov(WC99[msk].T)+1e-4*np.eye(192); Sia=np.linalg.solve(Sg,a); snr=float(a@Sia); w=Sia/snr
    pr=dQ[msk]@w; print('   SNR/unit^2 %.3f noise sd(dose units) %.3f | corr(w.dQ,r1) %.3f slope %.3f | trace S %.3f'%(snr,1/np.sqrt(snr),np.corrcoef(pr,Fm[:,col])[0,1],np.polyfit(Fm[:,col],pr,1)[0],np.trace(Sg)))
    res[nm]=dict(a=a,w=w,snr=snr,c0=float((WC99[msk]@w).mean()))
```