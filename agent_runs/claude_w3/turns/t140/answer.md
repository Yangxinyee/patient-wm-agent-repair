That `CFG` is only a small encoding config (GELU activation, unit codes), not the layer sizes. I'll print `VOCAB['config']` (the official sizes, possibly head counts) and the shapes of the encoder, head and z tensors. The rebuild needs the attention head count and the MLP widths. If the head count isn't listed, I'll infer it by testing candidate head counts against the saved states.

```python
print({k: v for k, v in VOCAB['config'].items()} if len(str(VOCAB['config'])) < 1500 else str(VOCAB['config'])[:1500])
sel_ = [n for n in NAMES if n.split('.')[0] in ('encode','encode_action','z_post','z_prior','z_in','z_head','gap','last_numeric','attr_heads','query_norm','type_emb','embed')]
print('; '.join(f"{n}{list(SHAPES[n])}" for n in sel_))
```