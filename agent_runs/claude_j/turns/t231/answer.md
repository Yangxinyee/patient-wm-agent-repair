Neither variant separates counterfactual from factual states well enough:
- **Ridge with observed-residual factual targets:** exposed factual rows still cost 0.069 nats per glucose reading.
- **Gradient boosting:** handles the unexposed rows much better (0.022) but is worse on exposed rows (0.080), at similar counterfactual recovery of about 70–75% of the needed shift.

So `with_code` carries the counterfactual signal only imprecisely. The budget allows a factual mean distortion of only about 1.8 mg/dL RMS on glucose rows.

Next I map the achievable trade-off on exposed rows alone, assuming unexposed rows get gated off. I'll use a ridge whose weights reflect each row's NLL sensitivity (tilt gain squared over variance for factual rows, gain squared for counterfactual rows) and sweep the factual weight.
- If some setting keeps the exposed factual cost near 0.01 per glucose reading while recovering at least 40% of the counterfactual shift, a gated head edit is worth building.
- If not, I'll look at the upstream input path instead.

```python
ixE=np.where(isG)[0]; hE=M.hour.values; stE=M.stay.values
EXP=np.array([h_>first[s_] for h_,s_ in zip(hE,stE)])
VAR_=np.maximum(GAIN*25,1.0)
def frontier(wf,alpha=100):
    msk=isG&EXP; w=np.where(NONE,wf*GAIN**2/VAR_,GAIN**2/VAR_.mean())
    ix,oof=oof_fit(msk,TH,w,lambda:Ridge(alpha=alpha)); mn=np.asarray(tilt_mean(torch.tensor(LG[ix]),list(EVV[ix]),vt,torch.tensor(oof)),float)
    f=NONE[ix]; return dnll_groups(ix,oof,vG,(('expF',f),('expF>12',f&(M.hour.values[ix]>12))))+' |'+summarize(ix,oof,mn,vG,GLU,'insulin').split('|')[1]
for wf in (10,50,200,1000): print('wf',wf,frontier(wf))
```