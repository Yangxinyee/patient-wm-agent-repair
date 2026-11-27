The decoder test shows the head's input carries recent glucose well: the policy dose at lag 1 has R² 0.96, at lag 2 0.92. Doses are carried more weakly: lag-1 insulin R² 0.69, lag-2 0.57, lag-3 0.52. More importantly, a plain least-squares decoder responds only 45% / 30% / 25% to a counterfactual dose change. That is attenuation, because it is fitted mostly on factual variation.

What the edit actually needs is the response to dose changes. So I'll fit a decoder directly on the state differences between each edited arm and the factual arm. Its target is the correction change needed at that reading: the structural one-step response Σc_k·δd_k (c derived from the records' kernel and AR) minus the model's own predicted-mean change. If these differences predict the target with high R², the head-level correction is viable and accurate for counterfactuals. I'll then check how noisy the same decoder is on factual rows, which bounds its NLL cost.

```python
Kv_i=np.array(Kins); a_d=np.array([0.468,0.322,0.195])
c_k=np.zeros(10)
for k in range(10):
    c_k[k]=Kv_i[k]-sum(a_d[j]*Kv_i[k-1-j] for j in range(3) if k-1-j>=0)
m_k=np.array([-0.52,-0.13,0.21,0.12,0.73,0.59,0.37,0.23,0.17,0.19])
print('c_k',c_k.round(2)); print('c-m',(c_k-m_k).round(2))
DL=np.stack([dose_lag(k)[0] for k in range(1,11)],1); DL0=np.stack([dose_lag(k)[1] for k in range(1,11)],1)  # arm doses, factual doses
PL=np.stack([pol_lag(k) for k in range(1,11)],1)
dpm=pmI-pm0
dC=(DL-DL0)@c_k - dpm                          # needed change in correction vs factual arm
Cf=(DL0-PL)@(c_k-m_k)                          # factual deviation correction
# map factual u for each row
idx0=pd.Series(np.arange(len(hnI))[armI==0],index=pd.MultiIndex.from_arrays([siI[armI==0],hjI[armI==0]]))
r0=idx0.reindex(pd.MultiIndex.from_arrays([siI,hjI])).values
dU=uI-uI[r0]
pert=glu_rows&(armI>0)&(np.abs((DL-DL0)).sum(1)>1e-9)
print('n pert rows',pert.sum(),'sd dC',dC[pert].std().round(2),'mean dC',dC[pert].mean().round(2),'sd Cf',Cf[glu_rows&(armI==0)].std().round(2))
for alpha in [0.1,1.0,10.0]:
    rg=Ridge(alpha=alpha,fit_intercept=False).fit(dU[pert&trn],dC[pert&trn])
    p_=rg.predict(dU[pert&tst]); r2=1-np.mean((dC[pert&tst]-p_)**2)/np.mean(dC[pert&tst]**2)
    sl=np.polyfit(dC[pert&tst],p_,1)[0]
    # apply to factual rows
    fr=glu_rows&(armI==0)
    pf=rg.predict(uI[fr]); pf=pf-pf.mean()
    print(alpha,'cf R2',round(r2,3),'slope',round(sl,3),'| factual: sd',round(pf.std(),2),'corr w Cf',round(np.corrcoef(pf,Cf[fr])[0,1],3))
```