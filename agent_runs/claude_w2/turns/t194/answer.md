The notes addendum is now written; the run stopped on its second line because `pf` is an array, not the policy function. My earlier notes call the policy function `pfun`, so I'll use it if it is callable. Otherwise I'll rebuild it from the training records with build_policy_table and make_pf_band. The experiment is unchanged: the unedited model versus the deviation oracle at kap = 0, 0.42 and 1, on 100 training stays and four insulin arms, against the record-based truth.

```python
if 'pfun' in globals() and callable(pfun): PF=pfun
else:
    _G,_tab=build_policy_table(tr); PF,_=make_pf_band(_tab)
print('policy', np.round(PF(np.array([140.,155,170,200,230,255,280,320])),2))
ids=list(ids_o); R=prep_rollout(tr, ids)
DGi=dose_stack(R, arms_o, 'insulin')
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