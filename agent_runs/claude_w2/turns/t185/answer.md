The factual data clearly support the correction. Regressing the model's teacher-forced glucose residuals (n = 34,711) on dose deviations gives −1.40, −1.49, −0.47, +0.52, +0.66, +0.40, +0.20, +0.50, +0.68, +0.40, with se ≈ 0.14. That is close to the filter G′ the rollouts need (−1.75, −1.54, −0.64, +0.06, +0.41, +0.50, …). So the model misses these effects in factual prediction too, and an exact deviation correction should gain roughly 0.005 nats per glucose reading.

Yet my oracle test on the 100-stay set showed a slight loss (+0.0009). That points to a bug or a mismatch in the oracle hook: deviation alignment, the tilt, or something specific to those stays. Next I debug it on the same stays. I compare the hook's correction r with the uncorrected residuals (slope of residual on r should be about 1) and check the NLL change as r is scaled. If the slope is about 1 but NLL still doesn't improve, the problem is the tilt or the bin NLL. If the slope is far from 1, the deviation computation is misaligned.

```python
hk=make_dev_oracle(Ro,DGi_o[:1],Gp,t_hide=1e9,pf=pfv)
cap=[]
def hk_cap(k,hn_,hj,P,ctx):
    A_=Ro['A']; code=A_.code.values; tt=A_.t.values
    m=np.isin(code[hj],GLC)
    if m.any():
        n=hn_[m]; si=ctx['seq_si'][n]; a=ctx['seq_arm'][n]; tau=np.round(tt[hj[m]]).astype(int)
        D=hk.devs(n,si,a,tau,ctx['drawn'],len(Gp))
        cen=torch.as_tensor(CENT[code[hj[m]]],dtype=torch.float32)
        cap.append((hj[m].copy(),D,(P[torch.as_tensor(np.where(m)[0])][:,1:]*cen).sum(1).numpy(),P[torch.as_tensor(np.where(m)[0])].numpy()))
    return P
rr=run_rollout_h(Ro,W,[{}],n_paths=1,seed=0,t_hide=1e9,want_nll=True,hook_P=hk_cap)
J=np.concatenate([c[0] for c in cap]); Dd=np.concatenate([c[1] for c in cap]); PMc=np.concatenate([c[2] for c in cap]); PP=np.concatenate([c[3] for c in cap])
A_=Ro['A']; vals=A_.value.values[J]; ok=~np.isnan(vals)
resid=vals-PMc; rcor=Dd@Gp
print('n',ok.sum(),'resid sd',resid[ok].std(),' r sd',rcor[ok].std(),' corr',np.corrcoef(resid[ok],rcor[ok])[0,1])
print('slope resid on r', np.polyfit(rcor[ok],resid[ok],1))
bb,se=ols(resid[ok],Dd[ok]); print('resid ~ dev lags (Ro):',np.round(bb,2),' se',np.round(se,2))
# NLL vs scale of r
code_=A_.code.values[J]; bf=np.array([1+np.searchsorted(np.array(NUMD[(CNAME[c],'value')]['fine_bin_edges']),v,side='left') if not np.isnan(v) else 0 for c,v in zip(code_,vals)])
for s_ in [0,0.5,1.0]:
    Pn=tilt_P(torch.as_tensor(PP),torch.as_tensor(CENT[code_],dtype=torch.float32),s_*rcor).numpy()
    nl=-np.log(Pn[np.arange(len(J)),bf]); print('scale',s_,'glucose NLL %.5f'%nl[ok].mean())
print('dev nonzero share',(np.abs(Dd[ok,0])>0).mean(),' mean dev lag1',Dd[ok,0].mean(),' sd',Dd[ok,0].std())
```