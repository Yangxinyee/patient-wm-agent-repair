On these 100 stays my kernel estimates give true readouts several times larger than the model's:

| edit | model | truth | ratio |
|---|---|---|---|
| insulin x2 | -2.75 | -13.95 | about 5 |
| insulin x1/2 | +1.38 | +6.98 | about 5 |
| norepinephrine x2 | +0.69 | +4.28 | about 6 |
| norepinephrine x1/2 | -0.58 | -2.14 | about 4 |

The rows are exactly the hours with a recorded glucose_finger, or map_arterial for norepinephrine, and the readout is the pooled mean over reached rows. The time course is also wrong. The true insulin x2 shift jumps to -3.5 at hour 13 and plateaus around -16 by hour 18. The model starts at -1.0 and creeps linearly to -4.1. So the model's one-hour sensitivity is about -0.57 per unit against -2.0, and its effect accumulates on a slower timescale.

A plain rescaling of the dose pathway would fix the average for start hour 12 but not the time course. The hidden readouts (start hours 6 and 16, x0 and x3) depend on getting both the size and the timing right. To design a principled edit I need to see inside the model, so I plan to reimplement the forward pass on the CPU and check it against the tool's states output.

First I need the embedding details the architecture note leaves open: which categorical rows are used, what `embed.kq` and `embed.text` are, and the inputs to last_numeric. Row norms will show which rows were trained. A simple row pattern, such as row 0 or a few rows per table, would make an exact reimplementation feasible.

```python
from safetensors import safe_open
for n in ['embed.kind.weight','embed.ekind.weight','embed.kq.weight','embed.text.weight','embed.categorical.status.weight','embed.categorical.unit.weight','embed.categorical.route.weight','embed.categorical.action.weight','embed.categorical.rate_unit.weight','embed.categorical.discharged_to.weight','type_emb.weight','last_status.weight']:
    w = load(n); print(n, tuple(w.shape), np.round(w.norm(dim=1).numpy(),3))
w = load('embed.numeric.weight'); print('numeric cols norms', np.round(w.norm(dim=0).numpy(),3))
w = load('last_numeric.weight'); print('last_numeric cols norms', np.round(w.norm(dim=0).numpy(),3))
for n in ['embed.bins.value.weight','embed.bins.dose.weight','embed.bins.rate.weight']:
    print(n, np.round(load(n).norm(dim=1).numpy(),2))
print('code norms', np.round(load('embed.code.weight').norm(dim=1).numpy(),2))
print('code_bias', np.round(load('code_bias').numpy(),2))
print('flow.a timescales (h):', np.round(np.sort(1/np.exp(load('core.flow.a').numpy())/3600),3)[:20], '...', np.round(np.sort(1/np.exp(load('core.flow.a').numpy())/3600)[-5:],1))
print(open('/model/ARCHITECTURE.md').read()[:0])
```