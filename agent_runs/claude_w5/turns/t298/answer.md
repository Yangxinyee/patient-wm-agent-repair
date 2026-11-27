The documentation matches what I simulate: the record through hour 12 as recorded, readings drawn from hour 13, every drug atom from hour 12 on scaled in both rate and dose, and the predicted mean averaged over 20 paths with z drawn from the prior four times.

So the gap between my local hour 13 (x1/2 +1.28) and the tool's (+0.09) must show up stay by stay. I'll compare per stay at hour 13:
- the tool's factual predicted mean (edited model) against my local factual prediction;
- the tool's x2 and x1/2 shifts against mine.

If the factual levels match but the x1/2 shifts don't, the cause is specific to the edited atoms, for example rate/dose bins or clipping in the tool. If even the factual levels differ, the tool's hour-13 prediction does not come from the state I think it does: different timing, z handling, or bundles in between.

```python
import numpy as np
t13 = d6e[(d6e.hour==13)].pivot_table(index='stay_id', columns='edit', values=['shift','factual'])
o13 = d6o[(d6o.hour==13)].pivot_table(index='stay_id', columns='edit', values=['shift','factual'])
cmpI = D13[D13.drug=='insulin'].set_index('sid').join(t13['shift'][['insulin_x2','insulin_half']]).join(t13['factual']['insulin_x2'].rename('tool_fact_e')).join(o13['factual']['insulin_x2'].rename('tool_fact_o'))
cmpI = cmpI.dropna()
print('insulin stays matched', len(cmpI))
print('factual level: corr(tool_e, loc_e) %.4f  mean diff %.3f  median |diff| %.3f' % (np.corrcoef(cmpI.tool_fact_e, cmpI.Ef)[0,1], (cmpI.tool_fact_e-cmpI.Ef).mean(), (cmpI.tool_fact_e-cmpI.Ef).abs().median()))
print('orig factual: mean diff %.3f  median |diff| %.3f' % ((cmpI.tool_fact_o-cmpI.Of).mean(), (cmpI.tool_fact_o-cmpI.Of).abs().median()))
print('x2: corr %.3f  tool mean %.3f loc mean %.3f | half: corr %.3f tool mean %.3f loc mean %.3f' % (np.corrcoef(cmpI.insulin_x2, cmpI.Ex2)[0,1], cmpI.insulin_x2.mean(), cmpI.Ex2.mean(), np.corrcoef(cmpI.insulin_half, cmpI.Ehalf)[0,1], cmpI.insulin_half.mean(), cmpI.Ehalf.mean()))
print(cmpI[['rate','Ef','tool_fact_e','Ex2','insulin_x2','Ehalf','insulin_half']].head(12).round(2).to_string())
```