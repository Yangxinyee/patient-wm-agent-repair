The binning matters a lot:
- **Insulin:** recorded rates map to fine bins 0.5→1, 1→4, 1.5→8, 2→10, 2.5→11, 3→12, 3.5→13, 4→14, 4.5–5→15, ≥5.5→16. Bins 2, 3, 5, 6, 7 and 9 never occur in training, and rate 0 shares bin 1 with 0.5.
- **Norepinephrine:** rates above 1.0 fall into bin 16, which was never seen in training. z_rate tops out at 0.915 in training, but ×2 of a 1.0 rate gives z = 2.6.

So the ×2 and ×3 norepinephrine edits send the model into untrained inputs, and ×0 insulin looks to the model like a very low dose.

For both drugs, z_dose equals or is an affine copy of z_rate. A clean fix would therefore make the dose response linear in z_rate and stop the bins from carrying dose magnitude.

How to implement and verify such a fix is the key question. A local CPU reimplementation of the forward pass, validated against the tool's `states` and `factual` outputs, would let me test edits without spending forward passes. First I'll check what extra detail `tensors.json` gives.

```python
tj = json.load(open('/model/tensors.json'))
print(type(tj), len(tj))
if isinstance(tj, dict):
    ks = list(tj.keys()); print(ks[:10])
    for k in ks[:5]: print(k, json.dumps(tj[k])[:600])
else:
    for x in tj[:5]: print(json.dumps(x)[:600])
```