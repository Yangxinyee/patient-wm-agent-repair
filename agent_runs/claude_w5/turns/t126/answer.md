Changing `age_mode` does not move the error at all. So the old `test_step` is probably an earlier variant that does not include the later fixes that gave the exact match. Rather than keep reverse-engineering it, I will write a fresh step function from the precise description in my notebook.

For that I need the exact tensor names and shapes of the embedding, encoders and core. Next I print them compactly, leaving out the head tensors.

```python
skip = ('attr_heads','time_head','count_head','monitor','delay','cont','exec','filter','pending','status','last_','query','type_emb','embed_transition','code_bias','z_head')
nm_core = [n for n in NAMES if not n.startswith(skip)]
print(len(nm_core))
print('; '.join(f"{n}{tuple(SHAPES[n])}" for n in nm_core)[:5000])
```