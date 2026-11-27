The generator ran for all 100 tuning stays in 88 s, and the rebound mod_events reproduces the old states exactly.

The counterfactual data confirm the core problem. On exact counterfactual histories (scaled doses and correspondingly shifted past readings) the model's teacher-forced prediction moves far less than the truth:
- insulin x2 from hour 12, GF: −1.64 against a true −12.85;
- insulin x0: +1.83 against +12.85;
- NE x2, MA: +1.34 against +4.18.

So the edit has to supply roughly 85% of the insulin effect and 70% of the NE effect.

Next step:
1. Assemble the data and choose tilt directions over the bins: centred GF centres for the glucose unit, MA centres for the MAP unit.
2. Solve per reading for the tilt θ that moves the predicted mean to (factual mean + true shift). Factual rows get target θ = 0.
3. Fit a ridge readout of θ on `with_code`, cross-validated by stay.

If the out-of-fold readout reproduces most of the mean shift on counterfactual rows (say above 80%) while staying near 0 on factual rows, the dead-unit plan is viable. A poor fit would mean `with_code` doesn't carry the information linearly, and I'd need richer features.

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
M=pd.concat([c[0] for c in CFD],ignore_index=True); WC=np.concatenate([c[1] for c in CFD]); LG=np.concatenate([c[2] for c in CFD])
print(M.shape, WC.shape, LG.shape); peek('fb_')
for c in GLU+MAPC: print(c, np.round(np.asarray(CEN[c])[[0,1,-2,-1]],1))
vG=np.zeros(17); vG[1:]=(np.asarray(CEN['glucose_finger'])-160)/25
vM=np.zeros(17); vM[1:]=(np.asarray(CEN['map_arterial'])-75)/8
isG=M.event.isin(GLU).values; isM=M.event.isin(MAPC).values
T=M.mean_f.values+np.where((isG&(M.drug=='insulin').values)|(isM&(M.drug=='norepinephrine').values),M.S.values,0.0)
TH=np.zeros(len(M)); t0=time.time()
for msk,v in ((isG,vG),(isM,vM)):
    ix=np.where(msk)[0]
    th=solve_theta(torch.tensor(LG[ix]),list(M.event.values[ix]),torch.tensor(v),torch.tensor(T[ix]))
    TH[ix]=np.asarray(th,dtype=float).ravel()
print('theta sec %.0f'%(time.time()-t0), 'G q01/50/99', np.round(np.quantile(TH[isG],[.01,.5,.99]),2),'M', np.round(np.quantile(TH[isM],[.01,.5,.99]),2), 'at bound', int((np.abs(TH)>7.9).sum()))
ix=np.where(isG)[0][:2000]; chk=np.asarray(tilt_mean(torch.tensor(LG[ix]),list(M.event.values[ix]),torch.tensor(vG),torch.tensor(TH[ix])),dtype=float).ravel()
print('tilt check maxerr %.3f'%np.abs(chk-T[ix]).max())
```