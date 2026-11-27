Almost every θ sat at the ±8 bound and the tilt check missed by up to 139. solve_theta's local variables (`base`, `tgt`) show that T is the target shift from the untilted mean, not the target mean itself.

I'll redo the solve with shift targets: (factual mean + true shift) − counterfactual mean, which is 0 for factual rows. I'll check the result against both readings of tilt_mean's return value (mean or shift). With correct inputs, the θs should mostly be well inside the bounds and the check error should be under 0.1.

```python
Tsh=T-M.mean_cf.values
TH=np.zeros(len(M)); t0=time.time()
for msk,v in ((isG,vG),(isM,vM)):
    ix=np.where(msk)[0]
    th=solve_theta(torch.tensor(LG[ix]),list(M.event.values[ix]),torch.tensor(v),torch.tensor(Tsh[ix]))
    TH[ix]=np.asarray(th,dtype=float).ravel()
print('sec %.0f'%(time.time()-t0),'G q01/50/99',np.round(np.quantile(TH[isG],[.01,.5,.99]),2),'M',np.round(np.quantile(TH[isM],[.01,.5,.99]),2),'at bound',int((np.abs(TH)>7.9).sum()))
ix=np.where(isG)[0][::20]; chk=np.asarray(tilt_mean(torch.tensor(LG[ix]),list(M.event.values[ix]),torch.tensor(vG),torch.tensor(TH[ix])),dtype=float).ravel()
print('chk vs mean target %.3f | vs shift target %.3f'%(np.abs(chk-T[ix]).max(),np.abs(chk-Tsh[ix]).max()))
c0=np.asarray(tilt_mean(torch.tensor(LG[ix]),list(M.event.values[ix]),torch.tensor(vG),torch.tensor(np.zeros(len(ix)))),dtype=float).ravel()
print('theta0 mean vs mean_cf %.3f'%np.abs(c0-M.mean_cf.values[ix]).max())
```