Run 83 shows the core's attention layers use bias-free q/kv/out projections. With 4 heads, the head dimension is 48. The recurrent blocks have three norms (self-attention, cross-attention to memory, FFN), while the prelude and coda have two. The memory "age" embedding is a small 1→192→192 MLP. The z-posterior, prior, gap and head tensors are under other prefixes, so I'll list the remaining tensors next.

```python
show([p for p in sorted({n.split('.')[0] for n in NAMES}) if p not in ('embed','encode','encode_action','core','embed_transition')])
```