The config fixes several unknowns: 4 attention heads (48 dimensions each), FFN width ×4, one prelude layer, two recurrent layers, one coda layer, 4 slots, memory 128 and z dimension 32. There is also an undocumented n_time_mix = 4. The vocabulary has no unit-index table, so the unit indices for the categorical embeddings (6 unit rows, 4 rate-unit rows) remain the main encoder uncertainty. Next I'll read what /data/README.md says about units, and print the distinct rate and dose units in the record. With the actual unit strings I can make an ordered guess (padding, unknown, then units by first appearance or alphabetically) and test it against the z posterior.

```python
rd = open('/data/README.md').read()
import re
for mt in re.finditer(r'unit', rd):
    pass
idx = [m_.start() for m_ in re.finditer('unit', rd)]
print(len(rd), idx[:10])
print(rd[:1200])
print('rate units', train.rate_unit.dropna().unique(), 'dose units', train.dose_unit.dropna().unique())
```