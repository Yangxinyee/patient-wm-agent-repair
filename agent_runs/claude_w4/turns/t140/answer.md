`UB` has norm 85, far larger than a LayerNorm output (about 14 × gain), so it is not the no-action u_k vector. My notes say that vector is LN(mlp2(size.bias)), the action encoder applied to an empty set. I'll list the encode_action tensors and compute it directly, so I can build the full token set at reading bundles.

```python
ea = [n for n in NAMES if n.startswith('encode_action')]
print([(n, SHAPES[n]) for n in ea])
```