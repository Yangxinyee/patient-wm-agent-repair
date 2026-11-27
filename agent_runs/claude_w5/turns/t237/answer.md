The early notebook has the true effects. Insulin per 1 u/h for one hour has a lagged effect on finger glucose: about −2 at lags 1–3, decaying to about 0 by lag 12, −12.7 cumulative. NE per 1 mcg/kg/min gives +3.2, +2.8, +1.6, +0.8, … on MAP. Dose noise is exogenous, so these are causal, and the rollout truths are about −14.3 / +7.15 / +4.11 / −2.05.

Next I'll break the fwd_4 shifts down by hour.
- If the edited-minus-original shift at hour 13 is already wrong-signed, the injection itself misfires under the tool's dose scaling. That would mean my local emulation of the scaled drug atoms, or the decoder, is off.
- If hour 13 is right and later hours reverse, the problem is rollout dynamics: c_hat soaking up the scaled earlier doses, or the model's reversion to the mean.

```python
m = de_o.merge(de_e, on=['stay_id','hour','edit'], suffixes=('_o','_e'))
m = m[m.reached_o]
tab = m.groupby(['edit','hour']).agg(n=('shift_o','size'), so=('shift_o','mean'), se=('shift_e','mean'),
                                     fo=('factual_o','mean'), fe=('factual_e','mean')).reset_index()
tab['d'] = tab.se - tab.so
for ed in ['insulin_x2','insulin_half','ne_x2','ne_half']:
    t = tab[tab.edit==ed]
    print(ed); print(t[['hour','n','so','se','d','fo','fe']].round(3).to_string(index=False))
# per stay at hour 13 for insulin_x2: correlation of edited shift with recorded rate at hour 12
```