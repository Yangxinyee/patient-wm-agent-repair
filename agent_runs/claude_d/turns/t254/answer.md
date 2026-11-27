Results on sel1 with E4:

| Readout | Edited | Original | Target |
|---|---|---|---|
| insulin x2 | −13.93 | −1.73 | −13.99 |
| insulin x1/2 | +7.03 | +0.91 | +6.99 |
| norepinephrine x2 | +1.03 | −0.08 | +4.85 |
| norepinephrine x1/2 | −2.15 | −0.48 | −2.43 |

Insulin is essentially on target, and norepinephrine x1/2 is close. Norepinephrine x2 is the problem.

The channel's added response is strongly asymmetric. Per unit of rate change it is about 1.1 for increases and about 4.3 for decreases, where a linear response would make them equal. So the model's representation of the norepinephrine rate saturates when the rate goes up.

My hypothesis is that doubling pushes many rates into the top fine bin, or beyond the trained range of the rate embedding. The state would then barely register the increase.

To check, I will look at:
- norepinephrine's rate bin edges and normalisation;
- how often doubling or halving changes the bin, and how often it lands in the top bin;
- the per-hour profile of the norepinephrine shifts in this pass.

If doubling mostly saturates the bins, an input-side fix becomes plausible. Because such rates are rare in the records, it should be nearly free in factual fit. One example is extending the top-bin embedding.

```python
print(list(VOCAB.keys()))
print(list(TRAIN.columns))
for k in VOCAB:
    v = VOCAB[k]
    if isinstance(v, dict) and 'norepinephrine' in v: print(k, str(v['norepinephrine'])[:300])
de = pd.read_csv('/work/fwd_11/dose_edit_edited.csv'); do = pd.read_csv('/work/fwd_11/dose_edit_original.csv')
print(list(de.columns)[:12], len(de))
```