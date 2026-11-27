The architecture shows that the value head reads `out`, the time-decayed `h`, `gap` and `z_head`, plus a per-code "last value" register. Dose information can therefore only reach the glucose and MAP predictions through the recurrent state.

My plan is a head-level edit. A shift is added to the glucose (and MAP) value logits, driven by a readout of recent dose deviations from the state. Because the head sits downstream of the state, factual states stay unchanged and I can check factual NLL offline.

First I need the exact head structure: the layers of `attr_heads.value_bin`, `query_norm`, `type_emb`, `last_numeric`, the `pending` pieces and `embed_transition`. If `value_bin` ends in a linear layer over 17 classes, a low-rank addition to that layer can tilt the predicted distribution.

```python
import json, re
groups = {}
for n in NAMES:
    g = n.split('.')[0]
    groups.setdefault(g, []).append(n)
for g, ns in groups.items():
    print(f"== {g} ({len(ns)})")
    if g in ('core',):
        # summarize core briefly
        print('   ', ', '.join(f"{n.replace('core.','')}{tuple(SHAPES[n])}" for n in ns if not n.startswith('core.blocks') and not n.startswith('core.prelude.0')))
        print('    [blocks/prelude tensors omitted]', sum(1 for n in ns if n.startswith('core.blocks') or n.startswith('core.prelude.0')))
    else:
        for n in ns:
            print('   ', n, tuple(SHAPES[n]))
print(VOCAB['config'])
```