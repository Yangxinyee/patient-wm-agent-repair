`heads.py` holds the view computation (out + out_norm(flowed h) + gap, and the z prior) plus reading-table helpers. The value-head function itself lives only in the kernel.

I will write a fresh, transparent value-head function, because I have to edit exactly those weights and the fitted readout must match how the model reads them. To do that I need the shapes of `last_numeric`, `last_status` and `type_emb`, and the length of the bin-centre lists. I will print them briefly.

```python
print({n:list(SHAPES[n]) for n in NAMES if n.startswith(('last_numeric','last_status','type_emb','query_norm','code_bias'))})
print({k:len(v) for k,v in CEN.items()}, CEN['glucose_finger'][:3], CEN['glucose_lab'][:3], CEN['map_arterial'][:3])
print(head_mean.__code__.co_consts[:20])
```