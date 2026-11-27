My reconstruction of `ctx` and the z prior is exact: rebuilt q matches the stored q with max error 0.0000. The first half of the prior output is the mean.

Now I can run the planned test. For each bundle that holds glucose or MAP readings:
- **z surprise:** posterior mean minus prior mean.
- **Regression:** z surprise on the bundle's glucose and MAP surprises, giving the directions v_G and v_M and how much of z they explain.
- **Propagation gain:** how strongly the model's next glucose (or MAP) prediction follows a surprise.

The plan is supported if R² is clearly positive and the gain is sizeable. That would mean writing "innovation × v_G" into z mimics a real surprise that the model carries forward. It is refuted if z encodes these surprises only weakly.

```python
ZPo_r=X2[bb,960:992]; Dz_r=ZPo_r-zpr[:,:32]
ub,first_i=np.unique(bb,return_index=True); pos_=np.searchsorted(ub,bb); Dz=Dz_r[first_i]
ev_=R['event'].values; sur=R['value'].values-R['mean_rec'].values
isGg=np.isin(ev_,['glucose_finger','glucose_lab']); isMm=np.isin(ev_,['map_arterial','map_cuff'])
sG=np.zeros(len(ub));hG=np.zeros(len(ub));sM=np.zeros(len(ub));hM=np.zeros(len(ub))
np.add.at(sG,pos_[isGg],sur[isGg]);np.add.at(hG,pos_[isGg],1);np.add.at(sM,pos_[isMm],sur[isMm]);np.add.at(hM,pos_[isMm],1)
sG=np.where(hG>0,sG/np.maximum(hG,1),0);sM=np.where(hM>0,sM/np.maximum(hM,1),0)
Dsg=np.column_stack([np.ones(len(ub)),hG>0,hM>0,sG,sM])
coefz=np.linalg.lstsq(Dsg,Dz,rcond=None)[0]; res_=Dz-Dsg@coefz
print('nb',len(ub),'R2 overall',np.round(1-res_.var(0).sum()/Dz.var(0).sum(),3),' mean sd Dz',np.round(Dz.std(0).mean(),3))
vG=coefz[3];vM=coefz[4]; print('|vG|',np.round(np.linalg.norm(vG),4),'|vM|',np.round(np.linalg.norm(vM),4),'cos',np.round(vG@vM/np.linalg.norm(vG)/np.linalg.norm(vM),3))
for nmv,v,m_,s_ in [('G',vG,hG>0,sG),('M',vM,hM>0,sM)]:
    u_=v/np.linalg.norm(v); pr=Dz@u_; print(nmv,'corr(Dz.u,s)',np.round(np.corrcoef(pr[m_],s_[m_])[0,1],3),'sd s',np.round(s_[m_].std(),2))
sid_=R['stay_id'].values; mr=R['mean_rec'].values
for evn in ['glucose_finger','map_arterial']:
    gi=np.where(ev_==evn)[0]; pr_=[(i,j) for i,j in zip(gi[:-1],gi[1:]) if sid_[i]==sid_[j]]
    a_=np.array([p[0] for p in pr_]); c_=np.array([p[1] for p in pr_])
    print(evn,'next-pred gain',np.round(np.polyfit(sur[a_],mr[c_]-mr[a_],1)[0],3),'n',len(a_))
```