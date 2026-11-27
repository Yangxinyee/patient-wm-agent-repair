Run 239 shows where the dose signal goes. A perturbation of u moves the internal state h a lot (0.68 per unit for a unit change in u, 45 for a change of 20). Yet `out`, which the heads read, barely moves (0.003 and 0.2). The state h has a huge norm (about 2,500), dominated by large accumulator dimensions, so the layer norm in front of `out` shrinks everything else about 90-fold.

An open question affects the targets. My notes say that on counterfactual histories the model's shift was only 13% of the truth, even though the past readings had been shifted by the true effect. That is surprisingly small if the model tracks its last reading. Next I will check whether the stored counterfactual rows really carry shifted readings, by comparing their observed values with the factual ones. If the readings were shifted, the doubled dose is actively cancelling the effect of the lower readings, which points to confounding in the model. If they were not shifted, my earlier targets were defined differently from what I believed.

```python
Fr=M[M.drug=='none'][['stay','b','event','obs','mean_f']]
Cr=M[(M.drug=='insulin')&(M.h0==12)&(M.m==2.0)][['stay','b','event','obs','S','mean_cf','mean_f','hour']]
JJ=Cr.merge(Fr,on=['stay','b','event'],suffixes=('','_F'))
g=JJ[JJ.event=='glucose_finger']; g=g[g.hour>=1]
print(len(g),'obs-obsF mean %.2f | S mean %.2f | obs-obsF-S %.2f | mean_cf-mean_f %.2f | mean_f-mean_fF %.3f'%((g.obs-g.obs_F).mean(),g.S.mean(),(g.obs-g.obs_F-g.S).mean(),(g.mean_cf-g.mean_f).mean(),(g.mean_f-g.mean_f_F).mean()))
print('corr(S_prev?)', np.corrcoef(g.S,g.mean_cf-g.mean_f)[0,1].round(3))
```