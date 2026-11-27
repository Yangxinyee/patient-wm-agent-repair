The policy function looks right: 0 below 150, about 1 U/h in 150–180, 2 in 180–220, 3 in 220–260, 3.9 in 260–300 and 6.4 above 300. The failure was mine: `eval_arms` was defined in the run that crashed on its second line, so it never existed.

Since kernel functions have no recoverable source after a restart, I'll put `eval_arms` in /work/evalh.py and exec it from there. The experiment is the same as before:
- the unedited model, and the deviation oracle at kap = 0, 0.42 and 1;
- 100 training stays, four insulin arms (x2, x1/2, x0, x2 from hour 16), 20 paths each;
- each compared with the record-based truth on the same rows, overall and at lags 1, 3, 6, 9 and 12 hours.

```python
code_eval='''
def eval_arms(res, arms, R, K=None, drug='insulin', outcome='glucose_finger'):
    K=Kins if K is None else K
    DG=dose_stack(R, arms, drug); outl=[]
    key={'insulin':'ins','norepinephrine':'ne'}[drug]
    for ai in range(1,len(arms)):
        t0=arms[ai].get(key+'_from',12.0)
        rows,reached,o=readout(R,res,arms,drug,outcome,t0=t0,lo=int(t0)+1,hi=24)
        sh=np.nanmean(o[ai]-o[0],1)[reached]; tr_=truth_rows(R,DG,ai,rows[reached],K)
        lag=np.round(R['A'].t.values[rows[reached]]).astype(int)-int(t0)
        prof=[(np.nanmean(sh[lag==l]), np.mean(tr_[lag==l])) for l in [1,3,6,9,12] if (lag==l).any()]
        outl.append((str({k:v for k,v in arms[ai].items() if not isinstance(v,np.ndarray)}), round(float(np.nanmean(sh)),2), round(float(np.mean(tr_)),2), len(sh), [(round(a,1),round(b,1)) for a,b in prof]))
    return outl
'''
open('/work/evalh.py','w').write(code_eval); exec(code_eval)
results={}
t0_=time.time()
res=run_rollout_h(R,W,arms_o,n_paths=20,seed=0,t_hide=13.0)
results['none']=eval_arms(res,arms_o,R); print('none',round(time.time()-t0_))
for r_ in results['none']: print('  ',r_)
for kp in [0.0,0.42,1.0]:
    Gf=G_from_K(Kins,kp); hk=make_dev_oracle(R,DGi,Gf,t_hide=13.0,pf=PF)
    res=run_rollout_h(R,W,arms_o,n_paths=20,seed=0,t_hide=13.0,hook_P=hk)
    results['dev_kap%.2f'%kp]=eval_arms(res,arms_o,R)
    print('kap',kp,'G',np.round(Gf,2),round(time.time()-t0_))
    for r_ in results['dev_kap%.2f'%kp]: print('  ',r_)
```