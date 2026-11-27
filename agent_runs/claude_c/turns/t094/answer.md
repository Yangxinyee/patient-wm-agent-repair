Run 82 showed the layout. The prelude has separate `attn.q` and `attn.kv` (K and V stacked). The action encoder has its own MLPs and final norm, and these touch only drug bundles. The value head is a 2-layer MLP on `with_code`.

My working view is that a useful dose amplifier has to sit in a path that only drug bundles use. Edits along state directions are shared by all bundles and would also scale natural variation many-fold, which would break the factual fit.

Before choosing between action-encoder levers and z_post/u-slice levers, I need the remaining core shapes (recurrent blocks, coda, flow/a_in/delta_in, z modules). I'll print them compactly.

```python
for n in NAMES:
    if n.startswith(('core.coda','core.c_out','core.slots','core.flow','core.a_in','core.delta_in','core.out_norm','core.blocks.0.0','z_','last_','embed.','gap','type_emb','query_norm')):
        print(n, SHAPES[n], end=' | ')
```