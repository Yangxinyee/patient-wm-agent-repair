tensors.json holds only shapes, so I can't run the network in the sandbox. Anything I learn about the model's internals must come from the tool's states/factual outputs and the weights.

That shapes the strategy. Under this ICU's dosing (decisions depend only on recorded data plus independent noise), the rollout gives the right counterfactual if the model's conditional prediction of the next reading depends on past doses the way the data does, holding past readings fixed. In the rollout, the model then propagates the shift through its own drawn readings. So I need two kernels:
1. The data's conditional dose kernel c_l: the partial effect of I_{t−l} on the next reading, holding the readings history fixed.
2. The model's current direct dose sensitivity.

Next I'll estimate c_l (l = 0..5) for both drugs from the training records with a partially linear double-ML: residualise the next reading and each lagged dose on readings-only history, then a joint OLS. I'll also read the model's direct sensitivity off forward pass 1, from the shift at the first reached hour, before its own readings have fed back. If the data kernel looks like -2·0.6^l for insulin and the model's direct sensitivity is a small fraction of c_0, the correction I need is a large dose-history term in the glucose head.

```python
# model's direct (first reached hour) sensitivity from fwd_1
fr = fw[fw.reached].sort_values('hour').groupby(['stay_id','edit']).head(1).copy()
def first_dose(sid, ed):
    i=sid_index[sid]; D = ins_t if ed.startswith('insulin') else ne_t
    h0 = next(h for h in range(12,25) if D[i,h]>0); return h0, D[i,h0]
fr['h0'], fr['d0'] = zip(*[first_dose(s,e) for s,e in zip(fr.stay_id, fr.edit)])
fr['dI'] = fr.d0*np.where(fr.edit.str.endswith('x2'),1.0,-0.5)
fr['lag'] = fr.hour - fr.h0
print(fr.groupby(['edit']).apply(lambda g: pd.Series({'n':len(g),'lag1_frac':(g.lag==1).mean(),'mean_shift':g['shift'].mean(),'mean_dI':g.dI.mean(),
      'slope_lag1': (g[g.lag==1]['shift']*g[g.lag==1].dI).sum()/(g[g.lag==1].dI**2).sum()})).round(3))
```