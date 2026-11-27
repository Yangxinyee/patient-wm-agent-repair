The writer works offline as designed:
- **Drug bundles:** the z shift tracks λ·innovation closely (correlation −0.992). The small residual comes from the rate read-out errors.
- **Reading bundles:** the shift is zero to within 3e-6.
- **Linear regime:** all three repurposed units stay at pre-activations of 14 to 27 on drug bundles.

The edit is saved, starting from the original weights with the earlier writer discarded.

**Next step: measure the effect with a factual forward pass on both models** for the 99 train stays whose states I have. This gives two things:
- the change in value NLL;
- the predicted-mean change at every reading. I can regress that on the lagged innovations to get the realised kernel per lag.

The design works if the lag-1 response is clearly negative for glucose and positive for MAP, scales with the gain, and NLL does not get worse. If the realised gain is tiny, z transport is weak and I'll scale λ up.

To write the request I need the stay ids, so I'll print them compactly.

```python
import json
ids99=[int(s) for s in np.unique(S2['stay_id'])]
print(len(ids99)); print(json.dumps(ids99))
```