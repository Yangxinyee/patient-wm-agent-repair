The model under-uses doses even in teacher-forced prediction:
- **Insulin:** the predicted next glucose_finger moves −0.45 mg/dL per U/h within band, against −1.7 observed in these stays and −2.0 in the full training data. Its residuals (observed minus predicted) fall by −1.26 mg/dL per U/h of dose.
- **NE:** the predicted MAP moves +0.94 per unit, against +3.3 observed.

So the factual conditional itself is attenuated. An edit that restores the true dose sensitivity should improve the factual NLL, not threaten it.

To design such an edit well, I want to replicate the model locally. Then I can measure how internal quantities depend on dose, test candidate edits on many training stays, and run my own rollouts without spending forward passes. The states tool lets me check pieces one at a time. For example, for a bundle that holds only an action, the z posterior mean depends on ctx (computable from the previous state), e_k and u_k. Matching it would confirm my atom embedding, action encoder and z_post.

First I'll inspect key tensors to infer conventions:
- the units of the time decay (core.flow.a, seconds or hours);
- the size and shape of the embedding tables;
- how the 8 inputs of last_numeric are used.

```python
open('/work/notes.md','a').write("""
## Model diagnostics (fwd 1-2, run 25-27), tuning sel1 (100 stays, /work/sel1.json)
- Original dose_edit: ins x2 -2.14, x1/2 +1.26, NE x2 +1.08, x1/2 -0.73.  My record-based truth on same reached rows: -14.1, +7.0, +4.7, -2.4.
- Model shift grows ~linearly with lag (-0.55 lag1 -> -3.5 lag12); truth -3.0 -> -19 (concave).
- Factual baseline value NLL 1.52793 (5238 values). Model factual within-band slope: insulin -0.45/U/h (data -2.0), NE +0.94 (data 3.6) => model under-uses doses even teacher-forced; amplifying dose effect should improve NLL.
- Readout rows = recorded outcome readings (gf / map_arterial) at hours 13-24, 'reached' = after a drug record at hour>=12.
""")
import torch
fa_ = load('core.flow.a'); print('flow.a exp(-a) range (s?)', np.exp(-fa_.numpy()).min(), np.exp(-fa_.numpy()).max())
for nm in ['embed.kind.weight','embed.ekind.weight','embed.categorical.unit.weight','embed.categorical.rate_unit.weight','embed.categorical.route.weight','embed.categorical.action.weight','embed.categorical.status.weight','embed.text.weight','embed.kq.weight','embed.bins.dose.weight','embed.bins.rate.weight','embed.bins.value.weight','embed.code.weight']:
    t = load(nm).numpy(); print(nm, t.shape, 'row norms', np.round(np.linalg.norm(t,axis=1),2))
nw = load('embed.numeric.weight').numpy(); print('numeric col norms', np.round(np.linalg.norm(nw,axis=0),3))
lw = load('last_numeric.weight').numpy(); print('last_numeric col norms', np.round(np.linalg.norm(lw,axis=0),3))
print('delta_in softplus range', np.log1p(np.exp(load('core.delta_in').numpy())).min(), np.log1p(np.exp(load('core.delta_in').numpy())).max())
print(json.load(open('/model/tensors.json')).__class__, list(json.load(open('/model/tensors.json')).items())[:3])
```