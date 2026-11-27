Recovered state so far:

- **Earlier analysis.** My data analysis is in effects.json: the insulin impulse response beta_ins, and the NE effect on MAP gam_ne (about +3.2, +2.6, +1.8, +0.9, +0.5 mmHg per unit of NE rate at lags 1–5).
- **Forward pass 2.** It was a states dump of 3 tuning stays, 156 bundles.
- **What runs 24–38 were doing.** Variables like `target_z1`, `first_err`, `variants` and `best` show I was trying to reproduce the model's first-step z-posterior on CPU from the weights. That never matched exactly: the best error was about 2.9.

Rather than push that exact re-implementation further, I want to rethink which weights can carry an edit. All dose information reaches the glucose prediction only through the state (out, h, z), which is read by the value head. Next I'll list every tensor name and shape, so I can see the structure of the heads (attr_heads.value_bin, query_norm, type_emb, embed_transition) and of the action encoder.

```python
tj = json.load(open('/model/tensors.json'))
for n in NAMES:
    print(n, SHAPES[n])
print(VOCAB.keys())
print(VOCAB['config'])
```