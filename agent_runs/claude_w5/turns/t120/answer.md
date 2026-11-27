No h dimension is dead: the smallest per-dimension spread is about 3, against a median of about 9. Repurposing a dimension will therefore disturb the model somewhat. I will have to choose by measured importance rather than find a free one.

How I wire the dose memory into the value head depends on whether that head is linear (shared across codes) or an MLP (which could separate glucose from MAP). Next I list the shapes of the head tensors.

```python
ks = [n for n in NAMES if any(t in n for t in ('attr_heads','query_norm','type_emb','z_head','embed_transition','out_norm','last_numeric'))]
for n in ks: print(n, SHAPES[n])
```