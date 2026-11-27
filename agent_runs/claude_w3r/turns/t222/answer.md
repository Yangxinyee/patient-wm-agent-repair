Neither table says which read it favoured, so I will stop trying to recover old results. I will rebuild the channel cleanly in three rank-1 stages:
- Stage 1: an edit of the prelude attention out-projection writes the insulin (or norepinephrine) rate, read from the attention outputs, into a quiet direction of the slot tokens.
- Stage 2: an edit of `core.b_in` carries that component into h along `a_y`.
- Stage 3: two unused (dead) value-head units, active only on that drug's reading code, turn it into a mean shift.

Calibration then uses forward passes. First I need the exact names and shapes of the tensors involved.

```python
pat=['prelude.0.attn','b_in','c_out','value_bin','embed_transition','out_norm','prelude_norm','core.slots','flow','query','type_emb','embed.code']
for n in NAMES:
    if any(p in n for p in pat): print(n, SHAPES[n])
```