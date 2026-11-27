Run 209 confirms the helpers exist: `ctx_of(row_prev, dt)`, the posterior pieces, and the gap MLP `gap.*`.

Now the key test of the new plan. Over the readings in the 99 stays I will compute:
- **z surprise at each reading's bundle:** z posterior mean (from `S2`) minus z prior mean, the prior recomputed from `ctx_of(previous row, dt)`.
- **Regression:** that z surprise on the bundle's glucose surprise and MAP surprise (value minus the tool's predicted mean), giving the directions v_G and v_M.
- **Propagation gain:** how much a glucose surprise moves the model's next glucose prediction.

The plan is supported if z surprise is explained to a substantial degree, with R² of a few tenths along those directions, and if surprises propagate with a sizeable gain. Then writing "dose innovation × v_G" into z mimics a real surprise that the model carries forward. It is refuted if z barely encodes the surprises; I would then write through the prelude attention instead.

```python
X2=S2['x']; st2=S2['stay_id']; hr2=S2['hour']
Rk=R['k'].values.astype(int); bb=Rk+1; dtr=R['dt'].values
okk=(st2[bb]==R['stay_id'].values)&(np.abs(hr2[bb]-R['hour'].values)<1e-6); print('align',okk.mean())
ub,first_i=np.unique(bb,return_index=True)
c0=ctx_of(X2[Rk[0]],dtr[0]); c0=np.asarray(c0,dtype=np.float64).ravel(); print('ctx shape',c0.shape)
CT=np.stack([np.asarray(ctx_of(X2[b_-1],dtr[i_]),dtype=np.float64).ravel() for b_,i_ in zip(ub,first_i)])
W0=load('z_prior.0.weight').numpy();b0=load('z_prior.0.bias').numpy();Wp2=load('z_prior.2.weight').numpy();bp2=load('z_prior.2.bias').numpy()
ZPr=gelu_np(CT@W0.T+b0)@Wp2.T+bp2
ZPo=X2[ub,960:992]
for half in [0,1]:
    cc=[np.corrcoef(ZPo[:,j],ZPr[:,32*half+j])[0,1] for j in range(32)]; print('half',half,'median corr',np.round(np.nanmedian(cc),3))
Dz=ZPo-ZPr[:,:32]
ev_=R['event'].values; sur=R['value'].values-R['mean_rec'].values
isGg=np.isin(ev_,['glucose_finger','glucose_lab']); isMm=np.isin(ev_,['map_arterial','map_cuff'])
pos_=np.searchsorted(ub,bb)
sG=np.zeros(len(ub));hG=np.zeros(len(ub));sM=np.zeros(len(ub));hM=np.zeros(len(ub))
np.add.at(sG,pos_[isGg],sur[isGg]);np.add.at(hG,pos_[isGg],1);np.add.at(sM,pos_[isMm],sur[isMm]);np.add.at(hM,pos_[isMm],1)
sG=np.where(hG>0,sG/np.maximum(hG,1),0);sM=np.where(hM>0,sM/np.maximum(hM,1),0)
Dsg=np.column_stack([np.ones(len(ub)),hG>0,hM>0,sG,sM])
coefz,_,_,_=np.linalg.lstsq(Dsg,Dz,rcond=None); res_=Dz-Dsg@coefz
print('R2 overall',np.round(1-res_.var()/Dz.var(),3),' sd Dz',np.round(Dz.std(0).mean(),3))
vG=coefz[3];vM=coefz[4]; print('|vG| per mg/dL',np.round(np.linalg.norm(vG),4),' |vM| per mmHg',np.round(np.linalg.norm(vM),4))
for nmv,v in [('G',vG),('M',vM)]:
    u_=v/np.linalg.norm(v); pr=Dz@u_; m_=(hG>0) if nmv=='G' else (hM>0); print(nmv,'corr(Dz.u, s)',np.round(np.corrcoef(pr[m_],(sG if nmv=='G' else sM)[m_])[0,1],3))
gi=np.where(R['event'].values=='glucose_finger')[0]; sid_=R['stay_id'].values
nx=[(i,j) for i,j in zip(gi[:-1],gi[1:]) if sid_[i]==sid_[j]]
ii_=np.array([a for a,_ in nx]);jj_=np.array([c for _,c in nx])
dp=R['mean_rec'].values[jj_]-R['mean_rec'].values[ii_]; gain=np.polyfit(sur[ii_],dp,1)[0]; print('glucose next-pred gain per surprise',np.round(gain,3),'n',len(ii_))
mi_=np.where(R['event'].values=='map_arterial')[0]; nx=[(i,j) for i,j in zip(mi_[:-1],mi_[1:]) if sid_[i]==sid_[j]]
ii2=np.array([a for a,_ in nx]);jj2=np.array([c for _,c in nx]); dp2=R['mean_rec'].values[jj2]-R['mean_rec'].values[ii2]; print('map next-pred gain',np.round(np.polyfit(sur[ii2],dp2,1)[0],3),'n',len(ii2))
```