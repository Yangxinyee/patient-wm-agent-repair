Killing single h dimensions often helps rather than hurts:

- Dimension 155 lowers the factual NLL by 0.0105 per reading; dimension 100 by 0.0100.
- A quarter of all dimensions cost less than 0.0002 each, and 10% improve the NLL by more than 0.0013.
- The helpful dimensions are mostly very slow ones (timescales of tens to thousands of hours).

These dimensions seem to integrate something that hurts predictions within a stay. They are therefore the natural place for the dose-memory channels, and they bring an NLL margin with them.

**Caveat.** A single-dimension effect may not add up when several are killed together.

**Next step.** Kill groups of the cheapest dimensions together, to pick 8 dimensions that can hold four antisymmetric memory pairs (fast and slow channels for each drug).

**What would confirm the choice.** A group of 8 whose combined change is clearly negative, or at worst near zero, for both glucose_finger and map_arterial. If the effects do not add up and the combined cost turns positive, I would prefer smaller groups.

```python
cands = [155, 100, 128, 151, 144, 171, 52, 130, 99, 73, 9, 132, 105, 104, 122, 137]
combos = {
 'K2': [155, 100], 'K4': [155, 100, 128, 151], 'K6': [155, 100, 128, 151, 171, 52],
 'K8a': [155, 100, 128, 151, 144, 171, 52, 130], 'K8b': [155, 100, 128, 151, 171, 52, 130, 99],
 'K8c': [155, 100, 128, 151, 171, 130, 99, 73], 'K10': [155, 100, 128, 151, 144, 171, 52, 130, 99, 73],
 'K12': [155, 100, 128, 151, 144, 171, 52, 130, 99, 73, 132, 105]}
res_k = {}
for nm_, D in combos.items():
    d_ = tf_nll(kill_dims(PO, D)) - nll0
    res_k[nm_] = d_
    print(nm_, D, 'all %.5f' % d_.mean().item(), ' per code', {int(c_): round(d_[torch.tensor(codes_np == c_)].mean().item(), 5) for c_ in np.unique(codes_np)})
```