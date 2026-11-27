No residual dimension is close to unused. Every dimension has LayerNorm gains near 1, and usage scores run from 0.72 to 1.15 around a median of 0.91. So repurposing a dimension removes real information. The factual-fit cost of that has to be measured, not assumed.

To design the dose-carrying path precisely I still need the model's internal activations at insulin bundles. In particular I need E, the prelude output read by `core.b_in`. That requires a local reimplementation of at least the atom embedding and the action encoder. The `states` tool's z-posterior can validate it, since z_post reads u_k directly.

First I need the exact atom fields, so I will print the data README and a short summary of the vocab keys. Seeing fields that map onto kind, ekind, unit, route and rate_unit would make the embedding reconstructable.

```python
print(open('/data/README.md').read()[:3000])
print(list(VOCAB.keys()))
print({k_: (type(v_).__name__, len(v_) if hasattr(v_,'__len__') else v_) for k_,v_ in VOCAB.items()})
```