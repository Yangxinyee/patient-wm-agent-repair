The replica matches the tool's states only on the first two bundles of each stay (error about 4e-05). From bundle 5 on it is off by 0.1 to 2. It is therefore not usable as it stands. Most likely the memory attention or the drug or END atoms are wrong.

Rebuilding it is costly, so I will first check whether a head-level edit can do the job. Such an edit would add a dose-dependent tilt to the glucose and MAP value logits, computed from the state, and I would calibrate it with the tool's dose_edit runs. Whether that is possible depends on the structure of the value head. If it is an MLP with a hidden layer, a correction can act on one code only. If it is a single linear map, a tilt would move every code together.

I will print only the tensor names and shapes for the heads.

```python
keys_ = [n for n in NAMES if any(s in n for s in ['attr_heads.value', 'embed_transition', 'last_', 'query_norm', 'type_emb', 'code_bias', 'z_head', 'out_norm', 'core.gap', 'heads.gap'])]
for n in keys_:
    print(n, SHAPES[n])
print(len(keys_), 'of', len(NAMES))
```