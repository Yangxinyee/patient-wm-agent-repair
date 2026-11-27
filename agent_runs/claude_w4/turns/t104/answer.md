Mixed insulin+NE bundles are common: 87k of 430k drug bundles, about a third of insulin bundles.

The bigger problem is that u_k barely encodes the insulin rate. Even for insulin-only bundles (62 grid points in 192 dimensions, which should be exactly interpolable), a linear readout leaves rms error 0.55 units/h. The encoder maps different rates to nearly the same vector. The NE rate, by contrast, is cleanly encoded in NE-only bundles (rms 0.0002).

So no readout inside z_post can recover the insulin dose. I have to create a clean rate channel upstream, inside `encode_action`, using spare hidden units:
- mlp1 (per atom) would carry a linear rate signal and a presence signal for each drug;
- mlp2 would pass them through linearly into a few dimensions of u_k;
- LayerNorm would perturb the rest of u_k only slightly.

First I need the encoder's tensor layout and its dead or low-use hidden units.

```python
for nm_ in NAMES:
    if nm_.startswith('encode_action') or nm_.startswith('z_post'):
        print(nm_, SHAPES[nm_])
```