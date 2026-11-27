Only bundles with drug atoms show the small z-posterior mismatches. Measurement-only bundles match to 1e-6. So my replica of the action encoder is slightly off, but it is very close.

Before refining the replica further, I want to settle how the edit will be built. That depends on where the dose signal can enter the glucose and MAP value heads. Next I'll list all tensor names and shapes, so I can see the structure of the heads (`attr_heads.value_bin`, `embed_transition`, `query_norm`) and of the core.

```python
for n_ in NAMES:
    print(n_, tuple(SHAPES[n_]), end=' | ')
```