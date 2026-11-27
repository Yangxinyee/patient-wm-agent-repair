The record format answers how bundles are built:
- each infusion row has a rate, a dose and an end hour;
- an END atom closes each infusion an hour later, usually in the same minute as the next infusion record;
- the 0.167 bundles are the end-of-visit markers at hour 25.17.

My cost estimate for a readout-only edit came out against it. A lag-1 innovation decoder with R² of about 0.24 needs a gain whose added noise would cost roughly 0.006–0.007 nats of factual error, over the limit. So I need the network to carry a cleaner dose signal, which means understanding the core step.

The plan is to build a one-step replica validated against the returned states. The z posterior mean is a direct check on the encoders, because it is computed from context plus the bundle encodings. I'll start with the embedding tables: their shapes, and which rows are zero (padding) or unused. That fixes the index conventions for units, kinds and the `kq`/`text` terms.

```python
for n in ['embed.kind.weight','embed.ekind.weight','embed.kq.weight','embed.text.weight','embed.code.weight',
          'embed.categorical.unit.weight','embed.categorical.rate_unit.weight','embed.categorical.action.weight',
          'embed.categorical.route.weight','embed.categorical.status.weight','embed.categorical.discharged_to.weight',
          'embed.numeric.weight','core.age.0.weight','core.age.2.weight','core.slots','core.prelude_norm.weight']:
    if n in SHAPES:
        w=load_original(n).numpy()
        print(n, tuple(w.shape), 'row norms' if w.ndim==2 else 'norm', np.round(np.linalg.norm(w.reshape(w.shape[0],-1),axis=1)[:16],2))
print([n for n in NAMES if n.startswith('core.age') or n.startswith('core.slots') or n.startswith('core.prelude.')][:30])
print(str(VOCAB['codes'][8])[:200], str(VOCAB['numeric_note'])[:50])
print(tu.rate_unit.unique()[:6], tu.dose_unit.unique()[:6], tu.event.unique())
```