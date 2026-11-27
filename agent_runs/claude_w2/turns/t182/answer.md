`make_pf` failed because pandas returned a read-only array. I take a writable copy and run the same tests again: the policy check, factual NLL with exact deviations, and the two oracle rollouts.

```python
code_pf2 = r'''
def make_pf(tab,minc=30):
    xs=np.arange(0,601); m=tab['mean'].reindex(xs).copy(); c=tab['count'].reindex(xs).fillna(0)
    m[c.values<minc]=np.nan; m.iloc[:150]=0.0
    m=m.ffill().bfill().to_numpy(copy=True); m[:150]=0.0
    def pf(x):
        x=np.asarray(x,dtype=float); return np.interp(x,xs,m)
    return pf,m
'''
open('/work/policy.py','a').write(code_pf2); exec(code_pf2)
pfv,pftab=make_pf(tabpol)
for x in [149,150,165,179,180,200,219,220,249,250,262,275,299,300,320,400]: print(x,round(float(pfv(x)),2),end='; ')
print()
open('/work/oracle.py','a').write(code_dor); exec(code_dor)
for nm_,hk in [('none',None),('dev-oracle',make_dev_oracle(Ro,DGi_o[:1],Gp,t_hide=1e9,pf=pfv))]:
    rr=run_rollout_h(Ro,W,[{}],n_paths=1,seed=0,t_hide=1e9,want_nll=True,hook_P=hk)
    A_=Ro['A']; ks=np.array(list(rr['nll'].keys())); js=ks%10**7; vv=np.array(list(rr['nll'].values()))
    cc=A_.code.values[js]; ok=~np.isnan(A_.value.values[js])
    print(nm_,'NLL all %.5f  glucose %.5f  map %.5f  n=%d'%(vv[ok].mean(),vv[ok&np.isin(cc,GLC)].mean(),vv[ok&np.isin(cc,MPC)].mean(),ok.sum()))
for nm_,hk in [('diff-oracle',make_oracle(Ro,DGi_o,Gp)),('dev-oracle',make_dev_oracle(Ro,DGi_o,Gp,pf=pfv))]:
    res_o=run_rollout_h(Ro,W,arms_o,n_paths=20,seed=3,hook_P=hk); line=[]
    for a,(lo_,hi_,t0a) in zip([1,2,3,4],[(13,24,12),(13,24,12),(13,24,12),(17,24,16)]):
        rows,reached,o=readout(Ro,res_o,arms_o,lo=lo_,hi=hi_,t0=t0a)
        pred=np.nanmean(o[a]-o[0],1); tru=truth_rows(Ro,DGi_o,a,rows,Kf)
        line.append(f'a{a}: {np.nanmean(pred[reached]):6.2f}/{tru[reached].mean():6.2f}')
    print(nm_,' | '.join(line))
```