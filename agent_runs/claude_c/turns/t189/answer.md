The list shows no separate dose pathway into the value head. The levers are `encode_action` (drug bundles only), the prelude, `b_in`/blocks and the heads. Next I will print the shapes of `encode_action.*`, `core.prelude.*`, `core.b_in` and `attr_heads.value_bin`, to work out where a selective gain could go.

```python
for n in NAMES:
    if n.startswith(('encode_action','core.prelude','core.b_in','core.c_out','attr_heads.value_bin','core.delta','core.a_in')):
        print(n, SHAPES[n])
```