This run strengthens the case for a small insulin effect rather than ruling it out:
- **Simple estimator agrees:** finger change on log-jitter with glucose-bin fixed effects is also positive at every lag (k = 3, 5, 7: about +0.05 to +0.24 per u/h).
- **Time windows:** they bounce around (hours 6–9 negative; 10–13 and 14–17 positive). The pooled pattern is a small positive response at lags 3–7.
- **Implied ×2 readouts:** about +0.9 overall, +1.9 ± 0.7 for hours 12–23 and +0.2 ± 0.7 for hours 2–11. The two periods do not differ significantly, so I should not trust the late-window number on its own.

Both drugs now show roughly 2-sigma effects in the "worse outcome" direction: more insulin goes with slightly higher glucose, and more norepinephrine with slightly lower MAP. The original model's tiny responses have the same signs (+0.066 and −0.027 for ×2).

That pattern fits three readings, and I need to tell them apart:
1. **Real, small, paradoxical effects** that the model learned only weakly.
2. **Confounding by indication:** a hidden transient factor raises the dose and also worsens the later outcome. The iid, uncorrelated innovations argue against this, but a meal-like iid factor could still do it.
3. **Sampling noise.**

Next I run discriminating checks:
- **Glucose_lab outcome:** an independent measurement of the same glucose. A real effect should appear there too.
- **Cross-drug placebos:** insulin jitter → MAP and norepinephrine innovation → glucose. These should be about 0 if the effects are drug-specific. Similar-sized "effects" would point to a general confounding or noise problem.
- **Dose-response:** the effect within each sliding-scale base level, and the residual outcome by log-jitter bin, which should be monotone and roughly linear if real.

```python
t0=time.time()
# 2) lab glucose outcome for insulin jitter
RIL=build_resid(INS0, irows_pos, ifeats, GL)
L,Ls,ro,ros=window_stats(RIL,np.ones(len(RIL['si']),bool),AwI); print('INS->LAB  L',L[:8].round(2),'se',Ls[:8].round(2),'readout %.2f (%.2f)'%(ro,ros))
L,Ls,ro,ros=window_stats(RI,np.ones(len(RI['si']),bool),AwI); print('INS->FING L',L[:8].round(2),'se',Ls[:8].round(2),'readout %.2f (%.2f)'%(ro,ros))
# 3) placebos
featsI2=dict(ifeats); featsI2.update({'ga0':GA,'gc0':GC,'ga1':lag(GA,1),'cmA':cummean(GA)})
RIP=build_resid(INS0, irows_pos, featsI2, MAPc)
L,Ls,ro,ros=window_stats(RIP,np.ones(len(RIP['si']),bool),Aw); print('PLACEBO INS->MAP L',L[:8].round(2),'se',Ls[:8].round(2))
featsN2=dict(feats); featsN2.update({'gf0':GF,'gl0':GL,'gf1':lag(GF,1),'cmF':cummean(GF)})
RNP=build_resid(NE0, cont, featsN2, GF)
L,Ls,ro,ros=window_stats(RNP,np.ones(len(RNP['si']),bool),AwI); print('PLACEBO NE->GLU L',L[:8].round(2),'se',Ls[:8].round(2))
# 4) dose-response heterogeneity: Ybar = mean of residual outcomes lags 3..7
Yb=np.nanmean(RI['Ly'][2:7],axis=0); ep=RI['ep']; okY=~np.isnan(Yb)
bI=base_of(Gpol)[RI['si'],RI['ti']]
for bv in (1,2,3,4,6):
    m=okY&(bI==bv); print(' base',bv,'n',m.sum(),'slope %.3f'%((ep[m]*Yb[m]).sum()/(ep[m]**2).sum()),'se~%.3f'%(Yb[m].std()/ep[m].std()/np.sqrt(m.sum())))
jit=np.log(INS0[RI['si'],RI['ti']]/np.maximum(bI,1e-9))
qs=np.quantile(jit[okY&(bI>0)],[0,.2,.4,.6,.8,1])
for i in range(5):
    m=okY&(bI>0)&(jit>=qs[i])&(jit<=qs[i+1]); print(' jitter bin %d [%.2f,%.2f] mean Ybar resid %.3f (se %.3f)'%(i,np.exp(qs[i]),np.exp(qs[i+1]),Yb[m].mean(),Yb[m].std()/np.sqrt(m.sum())))
print('time',round(time.time()-t0,1))
```