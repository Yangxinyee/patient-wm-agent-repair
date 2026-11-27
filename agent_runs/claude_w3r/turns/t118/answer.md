The pattern is clear. Action bundles reproduce well (relative h error 0.02 to 0.06). Measurement bundles are consistently worse (0.06 to 0.13), even when they hold a single map_arterial reading. So the core step is roughly right, and the likely culprit is how I embed measurement values (bin side, z-score definition, unit row or the kq term), which feeds both the prelude token and memory.

Next I will print my fbin/zsc definitions and scan encoding variants for measurement atoms: unit row, kq on/off, bin side, z clipping. I score each by the error on measurement bundles.

- **Confirms:** one variant brings measurement-bundle error down to the action-bundle level.
- **Refutes:** none helps. Then the remaining error comes from time-gap handling.

```python
print('\n'.join(lines[5:10]))
print('vocab keys:', list(VOCAB.keys()))
print('norm entry map_arterial:', {k_:(VOCAB[k_].get('map_arterial') if isinstance(VOCAB[k_],dict) else None) for k_ in VOCAB if isinstance(VOCAB[k_],dict) and 'map_arterial' in VOCAB[k_]})
```