impl.py includes a `core_step` with cross-attention to a memory `M` that takes age embeddings, plus several `age_mode` variants. So in runs 120-133 I was probably checking the core step (h/out update) against the fwd_3 states (`OUT`, `HH`). The "none / unit / numbias / atom_noattr" variants were probably attempts to reconstruct the memory `M`, and an error of ~2–4 means the core step does not match yet. I need the exact definition of the memory and core step, so I'll re-read ARCHITECTURE.md, starting with its length and the core-related section.

```python
arch=open('/model/ARCHITECTURE.md').read()
print(len(arch))
i=arch.lower().find('memory'); print('first memory at',i)
print(arch[:9500])
```