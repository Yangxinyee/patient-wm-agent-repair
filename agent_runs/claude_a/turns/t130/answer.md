The kernel still holds the full offline replica: the parameter dict P, the batched core step, the head, teacher-forced evaluation (`tf_eval`/`tf_states`), the rollout (`rollout_stay`/`rollout_gen`), and the targets `psi_ins`/`psi_ne`. The notebook head restores the key data facts for this simulated ICU:
- Readings come on the hour and infusions start at h+5 min. So the reading bundle and the drug bundle are separate.
- Insulin is given iff the finger glucose at that hour is ≥150. The rate is a tier base (1, 2, 3 or 4 for glucose tiers [150,180), [180,220), [220,250), ≥250) times an iid random multiplier.
- The insulin response per unit/h is about −2.0, −2.3, −2.0, −1.6, −1.1, −0.9, −0.6, −0.4 over lags 1–8.

This tells me how to keep the factual fit while adding an insulin effect. Within a tier the multiplier is random, so a memory of (rate − expected rate for the glucose tier) is unconfounded. With a correct coefficient it should even improve the factual NLL, because the model currently ignores the multiplier. Under a dose edit it moves by the full dose change.

To build that edit I need the exact tensor names and shapes for every place that reads or writes h dims: the prelude, blocks, c_out, coda, z networks, the head MLP and the LN gammas. I don't have them in view after the restart. I also want to see how P stores tensors (torch vs numpy) and what W/Wt/TJ are. If they are copies of P, I can build edited parameter sets offline and test them with `rollout_stay`/`tf_eval` before saving anything.

```python
import collections
grp = collections.OrderedDict()
for nme in NAMES:
    pre = nme.split('.')[0] if not nme.startswith('core') else '.'.join(nme.split('.')[:2])
    grp.setdefault(pre, []).append(f"{nme}{tuple(SHAPES[nme])}")
for k_,v_ in grp.items():
    print(k_, '|', ' '.join(v_))
print('P type', type(next(iter(P.values()))))
for dn_ in ['W','Wt','TJ']:
    D_ = globals()[dn_]
    diff = [n_ for n_ in NAMES if not torch.equal(torch.as_tensor(D_[n_]).float(), torch.as_tensor(P[n_]).float())]
    print(dn_, 'differs from P in', len(diff), diff[:8])
print('psi_ins', np.round(psi_ins,2), 'psi_ne', np.round(psi_ne,2))
```