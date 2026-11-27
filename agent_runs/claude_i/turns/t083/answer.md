All tensor groups are now visible. The full state computation needs:
- the embedding tables;
- the encode and encode_action set encoders;
- the gap MLP, z_post and z_in;
- the core tensors listed above.

The value head needs z_prior, z_head, type_emb, query_norm, last_numeric, attr_heads.value_bin and embed_transition.value_bin (indexed by code×17 + last value bin). So re-implementing is bounded and doable. Before writing new code I'll check how far the old z-posterior search got. Next I'll print the configs I was testing (cfg0, cfg_), the variant and gap-option keys, and best_u. A z-posterior match close to exact would mean the encoder and context are already right, leaving only the core step to rebuild.

```python
print('cfg0', cfg0); print('cfg_', cfg_)
print('variants', list(variants.keys()) if isinstance(variants, dict) else variants)
print('gapopts', list(gapopts.keys()))
print('best_u', best_u[:12])
print('mads', mads)
print('res_', [ (r if not hasattr(r,'__len__') or len(str(r))<80 else str(r)[:80]) for r in res_][:16])
```