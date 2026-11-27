This grouping is far more informative than the overall numbers. The stored errors come from my last route hypothesis ("act_all"), so I read them with that in mind.

| Bundle type | e error | Reading |
|---|---|---|
| Insulin start alone | ~1e-6 | empty-set encoding is correct |
| Hospital admission; admission END atoms | ~1e-6 | encoder MLP and visit featurization are correct |
| Drug-only bundles | 1.57–2.31, constant | END atoms do belong in e |
| Measurement bundles | 0.5–2.3, varying | featurization of measurement atoms is wrong |

Under "end_to_e" the END-only error was small but nonzero (0.13), so the END atoms' features in e are also slightly off.

Next I'll test featurization variants for measurement atoms against the 11 true e projections, bundle by bundle:
- the bin rule for ties;
- swapped unit rows;
- the z-score definition.

First I need my current `fbin` and `zsc` rules.

```python
src=open('/work/setup.py').read()
i=src.find('def zsc'); print(src[i:i+700])
```