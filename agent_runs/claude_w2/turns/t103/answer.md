The tensor list confirms the layer structure in ARCHITECTURE.md:
- prelude and coda are pre-norm layers with self-attention and an FFN (norm1, norm3).
- The two recurrent blocks add cross-attention to the memory (norm2).
- Attention projections are bias-free (q 192, kv 384, out 192; 4 heads of 48).
- The value head is a 192→192→17 MLP plus a per-(code, last bin) transition table.

The z_prior, z_post and z_in tensors and the embed tables did not match my filter. Next I print all remaining tensor names, the data README and the vocab's numeric and atom-kind entries. Then I will write one clean implementation file, /work/impl.py, covering encoding, core step and head, so the code survives restarts; the earlier function sources are lost. If the remaining names include z_post, z_prior and z_in MLPs with the expected shapes (576 inputs to z_post), the step can be fully specified.

```python
shown=('core','encode','attr_heads.value_bin','query_norm','z_head','out_norm','gap','type_emb','embed_transition','last','filter','embed.code')
for nmx in NAMES:
    if not nmx.startswith(shown): print(nmx, SHAPES[nmx])
print(open('/data/README.md').read()[:3000])
print(VOCAB['atom_kinds'], VOCAB['event_kinds'], VOCAB['numeric_note'], VOCAB['value_bin_note'])
```