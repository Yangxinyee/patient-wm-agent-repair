The factual cost at gain 12 is far over budget: +0.017 nats overall. Glucose readings carry it (GF +0.035, GL +0.053), while MAP is neutral or slightly better.

My reading of the cause: the amplified rate signal shifts factual glucose predictions according to the dose level. Under the dosing policy, that level tracks the last glucose reading, which the model already uses. That part can be cancelled with a per‑(code, last‑value‑bin) logit offset in `embed_transition.value_bin`. The offset depends only on the last reading, so it leaves the dose‑change response mostly intact.

This run will:
1. Measure how much of the factual prediction shift the last bin explains.
2. Estimate the factual cost with the correction.
3. Choose a gain that targets about 0.003 nats.
4. Save the gain and the correction.
5. List 100 fresh tuning stays for out‑of‑sample validation.

If the last bin explains a large share of the shift, the corrected cost should drop a lot. If not, I fall back to a lower gain alone.

```python
import pandas as pd
fo=pd.read_csv('/work/fwd_11/factual_original.csv'); fe=pd.read_csv('/work/fwd_11/factual_edited.csv')
cols=list(fo.columns); pm=[c for c in cols if 'mean' in c.lower()][0]; key=['stay_id','hour','event']
mm=fo.merge(fe[key+[pm]],on=key,suffixes=('_o','_e')); mm['d']=mm[pm+'_e']-mm[pm+'_o']
ob=[c for c in cols if c.lower() in ('value','observed','obs','y')]
if ob: mm['y']=mm[ob[0]]
else:
    tu=pd.read_csv('/data/tuning.csv',usecols=key+['value']); mm=mm.merge(tu,on=key,how='left'); mm['y']=mm['value']
EVC={'glucose_finger':6,'glucose_lab':10,'map_arterial':7,'map_cuff':5}
mm=mm.sort_values(['stay_id','event','hour']); mm['prev']=mm.groupby(['stay_id','event'])['y'].shift(1)
mm['lb']=[0 if p!=p else int(fb_(EVC[e],p)) for e,p in zip(mm.event,mm.prev)]
REP={'glucose_finger':.0349,'glucose_lab':.0532,'map_arterial':.0029,'map_cuff':-.0063}
CORR={}; tot=0; ntot=0
for e,c in EVC.items():
    q=mm[mm.event==e]; s2=float(((q.y-q[pm+'_o'])**2).mean()); gb=q.groupby('lb')['d']
    mb=(gb.sum()+20*q.d.mean())/(gb.size()+20); fit=q.lb.map(mb)
    r2=1-((q.d-fit)**2).sum()/((q.d-q.d.mean())**2).sum()
    a_e=(((q.y-q[pm+'_e'])**2-(q.y-q[pm+'_o'])**2)/(2*s2)).mean()
    a_c=(((q.y-q[pm+'_e']+fit)**2-(q.y-q[pm+'_o'])**2)/(2*s2)).mean()
    cc=REP[e]*(a_c/a_e) if c in (6,10) else REP[e]
    if c in (6,10): CORR[c]=(mb,s2)
    tot+=cc*len(q); ntot+=len(q)
    print(e[-6:],'n',len(q),'d %.2f sd %.2f R2lb %.2f'%(q.d.mean(),q.d.std(),r2),'approx e %.4f c %.4f -> %.4f'%(a_e,a_c,cc))
C12=tot/ntot; km1=min(11,11*np.sqrt(0.003/max(C12,1e-6))); kk=1+km1
print('corrected cost k12 est %.4f -> k %.2f'%(C12,kk))
ET0=A(load_original('embed_transition.value_bin.weight')).copy()
def mk3(k,cs=1.0):
    mk2(k,1.0); et=ET0.copy(); f=(k-1)/11*cs
    for c,(mb,s2) in CORR.items():
        v=np.r_[0,np.asarray(CEN[c])-np.mean(CEN[c])]
        for b,val in mb.items(): et[c*17+int(b)]+= -f*val/s2*v
    save_edit('embed_transition.value_bin.weight',TT(et))
mk3(kk); print('edited',edited())
tid=sorted(pd.read_csv('/data/tuning.csv',usecols=['stay_id']).stay_id.unique()); S1=set(s1)
VAL=[int(x) for x in tid if x not in S1][:100]; print('val suffixes',' '.join(str(x-2020000000) for x in VAL))
with open('/work/notes.md','a') as f:
    f.write("- fwd10/11: targeted mk2(12,12) (RB useless: RA captures B too): A x2 -9.18 half 3.31, B x2 1.80 half -1.46; factual +0.017 (GF+.035 GL+.053 MA+.003 MC-.006). TOO COSTLY.\n- run293: correction = per (code,lastbin) logit tilt in embed_transition for GF/GL cancelling mean pred shift by last bin (fit fwd11). mk3(k). VAL = 100 tuning stays not in sel1.\n")
```