The DataFrame scan hit a frame without a stay_id column, so the notes addendum was never written. I'll redo it first in this run.

I now have what I need:
- **Records:** `tr` holds the training stays (30,000) and `tu` the tuning stays (400).
- **Oracle hook:** make_dev_oracle shifts the predicted mean of each glucose reading by Σ_k G_k·(dose_{t−k} − policy(glucose_{t−k})), using the glucose on the current path (drawn when hidden).
- **Readout:** `readout` gives per-arm predictions at the glucose_finger rows of hours 13–24, flagged by whether insulin had already run.

This run is the oracle calibration that was never recorded. On the 100 training stays ids_o, five insulin arms (recorded, x2, x1/2, x0, x2 from hour 16) and 20 paths, I'll compare:
1. the unedited model;
2. the model plus the deviation oracle, with the direct kernel G = G_from_K(Kins, kap) for kap = 0, 0.42 and 1.

The comparison is against the record-based truth Σ_k Kins_k·Δdose_{t−k} on the same rows, with the shift also broken down by hours since the edit to check the shape.

What would confirm the hypothesis: some kap makes all four insulin arms land near the truth (x2 about -14, x1/2 about +7, roughly linear in dose). Then that tilt is what I build into the weights. What would refute it: no kap fits all arms, or the shape stays linear-growing whatever kap is. That would mean the model's own propagation of drawn values must also be changed.

```python
open('/work/notes.md','a').write(notes_add)
print('pf callable', callable(pf), pf(np.array([140.,155,170,200,230,255,280,320])))
import time
ids=list(ids_o); R=prep_rollout(tr, ids)
DGi=dose_stack(R, arms_o, 'insulin')
def eval_arms(res, arms, R, K=Kins, drug='insulin', outcome='glucose_finger'):
    DG=dose_stack(R, arms, drug); outl=[]
    key={'insulin':'ins','norepinephrine':'ne'}[drug]
    for ai in range(1,len(arms)):
        t0=arms[ai].get(key+'_from',12.0)
        rows,reached,o=readout(R,res,arms,drug,outcome,t0=t0,lo=int(t0)+1,hi=24)
        sh=np.nanmean(o[ai]-o[0],1)[reached]; tr_=truth_rows(R,DG,ai,rows[reached],K)
        lag=np.round(R['A'].t.values[rows[reached]]).astype(int)-int(t0)
        prof=[ (np.nanmean(sh[lag==l]), np.mean(tr_[lag==l])) for l in [1,3,6,9,12] if (lag==l).any()]
        outl.append((str(arms[ai]), round(float(np.nanmean(sh)),2), round(float(np.mean(tr_)),2), len(sh), [(round(a,1),round(b,1)) for a,b in prof]))
    return outl
results={}
t0_=time.time()
res=run_rollout_h(R,W,arms_o,n_paths=20,seed=0,t_hide=13.0)
results['none']=eval_arms(res,arms_o,R); print('none',time.time()-t0_); 
for r_ in results['none']: print('  ',r_)
for kp in [0.0,0.42,1.0]:
    Gf=G_from_K(Kins,kp); hk=make_dev_oracle(R,DGi,Gf,t_hide=13.0,pf=pf)
    res=run_rollout_h(R,W,arms_o,n_paths=20,seed=0,t_hide=13.0,hook_P=hk)
    results['dev_kap%.2f'%kp]=eval_arms(res,arms_o,R)
    print('kap',kp,'G',np.round(Gf,2),time.time()-t0_)
    for r_ in results['dev_kap%.2f'%kp]: print('  ',r_)
```