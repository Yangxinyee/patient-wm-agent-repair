The listing shows the remaining core tensors (decay, `b_in`/`c_out` maps, coda layer). None of them is specific to action bundles. That leaves the action encoder as the only place to change the action signal without touching reading pathways.

Next I'll test whether removing the encoder's constant output offset (`encode_action.norm.bias` → 0) raises the share of the action token that varies with the rate attributes. If it does, all four dose-edit readouts should move away from the originals (−1.85, +0.78, −0.50, −0.60) toward larger magnitudes. If they stay put, the offset isn't what dilutes the signal.

```python
scale('encode_action.norm.bias', 0.0)
print(ops())
```