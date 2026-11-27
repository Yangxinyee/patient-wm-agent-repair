The kernel still has my reconstruction helpers: `u_rec`, `e_empty`, `e_read`, `view_from`, `LNnp`, and `setenc_my`. The prelude is one attention layer (q, kv, out, no biases listed) plus an FFN, with LayerNorms. Before deciding how to amplify the dose channel, I want the attention weights the four slot queries give the u_k and z tokens at drug bundles. If attention to the z token is moderate but its value change is small, a dedicated quiet-z-dim channel amplified through `z_in` can work. If attention is near zero, I'll need another route. First: shapes, head count and helper signatures.

```python
import inspect
for f in ['u_rec','e_empty','e_read','view_from','LNnp','setenc_my','datom','end_atom','uenc','build_writer']:
    try: print(f, inspect.signature(globals()[f]))
    except Exception as ex: print(f,'?')
for n in ['core.flow.a','core.slots','z_in.weight','core.prelude.0.attn.kv.weight','core.prelude.0.norm1.weight']: print(n, SHAPES[n])
print([ (i,l[:150]) for i,l in enumerate(L) if 'head' in l.lower() and ('attn' in l.lower() or 'attention' in l.lower() or 'heads,' in l.lower())][:6])
```