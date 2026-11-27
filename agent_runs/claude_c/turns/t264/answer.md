For insulin x1/2, N5 helps every band, including the high-dose stays: [+0.23, +2.61, +4.50] against truth [+3.2, +8.0, +12.4]. For x2 it collapses at high doses: [-2.19, -2.70, -0.98] against [-6.3, -16.0, -24.7]. So the main readout is limited by the doubled high doses pushing the dose code into a saturated region. Half moves the code back toward familiar values, so the potency itself is not the limit there. To design a code that doesn't saturate, I need the exact definition of the dose z-score: linear in rate or log-based. I'll read the insulin and norepinephrine normalisation entries in the vocab.

```python
nm=VOCAB.get('normalisation', VOCAB.get('normalization'))
print(list(VOCAB.keys())[:12]); print(type(nm))
print({k:nm[k] for k in nm if 'insulin' in k or 'norep' in k} if isinstance(nm,dict) else str(nm)[:600])
```