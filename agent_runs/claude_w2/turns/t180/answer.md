The run failed because my old policy function `pfun` is a lambda that depends on a global since overwritten. Its source is gone, as my notes warned. So I rebuild the policy expectation from the training records and save it to a file. For each stay-hour I take the fresh glucose (finger, else lab) and the insulin rate started that hour (0 if none), and tabulate the mean rate per mg/dL. I check that the band steps (150, ~180, ~220, ~250, ~300) appear. Then I rerun the two oracle tests: factual NLL with an exact deviation correction, and rollouts with the difference-based and deviation-based oracles.

```python
code_pf = r'''
def build_policy_table(df):
    g=df[df.event.isin(['glucose_finger','glucose_lab'])&(np.abs(df.hour-np.round(df.hour))<1e-6)][['stay_id','hour','event','value']]
    gp=g.pivot_table(index=['stay_id','hour'],columns='event',values='value',aggfunc='first')
    gsel=gp['glucose_finger'].where(gp['glucose_finger'].notna(),gp.get('glucose_lab'))
    ins=df[df.event=='insulin'][['stay_id','hour','rate']].copy(); ins['h']=np.floor(ins.hour+1e-6)
    ir=ins.groupby(['stay_id','h']).rate.sum()
    G=pd.DataFrame({'g':gsel}).reset_index(); G['rate']=ir.reindex(pd.MultiIndex.from_arrays([G.stay_id,G.hour])).fillna(0).values
    G['gi']=np.clip(np.round(G.g).astype(int),0,600)
    tab=G.groupby('gi').rate.agg(['mean','count'])
    return G,tab
def make_pf(tab):
    xs=np.arange(0,601); m=tab['mean'].reindex(xs); c=tab['count'].reindex(xs).fillna(0)
    # smooth within +-3 mg/dL weighted by counts, but keep step at 150
    num=(m.fillna(0)*c).values; den=c.values; out=np.zeros(601)
    for i in xs:
        lo,hi=max(0,i-3),min(600,i+3)
        if i<150: hi=min(hi,149)
        else: lo=max(lo,150)
        out[i]=num[lo:hi+1].sum()/max(den[lo:hi+1].sum(),1)
    out[:150]=0.0
    def pf(x):
        x=np.asarray(x,dtype=float); return np.interp(x,xs,out)
    return pf,out
'''
open('/work/policy.py','w').write(code_pf); exec(code_pf)
Gpol,tabpol=build_policy_table(tr)
pfv,pftab=make_pf(tabpol)
print('n obs',len(Gpol),' rate>0 share',(Gpol.rate>0).mean())
for x in [140,149,150,155,165,175,179,180,185,200,219,220,230,249,250,260,274,275,290,299,300,320,400]: print(x,round(float(pfv(x)),3),end='; ')
print()
# residual sd of policy noise
Gpol['p']=pfv(Gpol.g.values); print('policy resid sd (g>=150):',(Gpol.rate-Gpol.p)[Gpol.g>=150].std())
exec(open('/work/oracle.py').read())
b_model=np.array([-0.26,-0.11,-0.06,-0.02,-0.01,-0.01,0.01,0,0,0,-0.01,0])
Gp=G_from_K(Kf,0.42)-b_model; print('Gprime',np.round(Gp,2))
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