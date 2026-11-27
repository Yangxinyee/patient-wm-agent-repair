The core blocks are standard. Every attention layer has q, kv and out projections with no biases. Each layer is pre-norm, with norm1 before attention, norm2 before cross-attention in the blocks, and norm3 before a 768-wide FFN. The small MLPs for gap, age, z_prior and z_post have the shapes the architecture describes. That makes a local reimplementation feasible. Two unknowns remain: the head count and the activation, and I can settle both by checking against the fwd_3 states.

The run stopped on a wrong tensor name (`embed.categorical.unit` needs its `.weight` suffix), so I didn't see the embedding details. Next I'll print:
- the exact embedding tensor names and shapes;
- the vocab keys and config;
- a short excerpt of the data README.

That shows how atoms are tokenised: categorical fields, kinds, bins and z-score normalisation. If the README columns map clearly onto the embedding fields, the reimplementation is viable. If they don't, I'll fall back to a head-only edit driven by the states.

```python
print([(x,SHAPES[x]) for x in NAMES if x.startswith('embed') or x.startswith('last_') or x.startswith('type_emb')])
print(list(VOCAB.keys()))
print(VOCAB['config'])
rd = open('/data/README.md').read(); print(len(rd)); print(rd[:2500])
```