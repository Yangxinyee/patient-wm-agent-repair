`pfun` is callable, but it is a lambda written for a pandas Series, so it fails on a plain array. I'll build the policy function fresh from the training records with `build_policy_table` and `make_pf_band`. These map the latest glucose to the mean insulin rate, keeping the step at 150 and the band boundaries. I'll print a few values to check it matches the sliding scale in my notes (0 below 150, about 1 U/h in 150–180, about 2 in 180–220, and higher above). The oracle experiment is otherwise unchanged.

```python
_G,_tab=build_policy_table(tr); PF,_pfarr=make_pf_band(_tab)
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