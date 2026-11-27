Full exact replica of the core step: gelu activations in the core FFNs and the age MLP, memory M that includes the current bundle's e, ages in seconds, and the age embedding added to the memory vectors before the key/value projection. h matches to a median 5e-5 and out to 5e-7. I'll record this in the notes now so it survives a restart. Then I'll list the head tensors so I can replicate the value head and check it against the per-reading predicted means from fwd_4.

```python
with open('/work/notes.md','a') as f:
    f.write("""
## Restart 4 (runs 142-153): EXACT REPLICA ACHIEVED
- z_post: zparts() exact (z = first 32 of mlp(z_post,[ctx,e,u]), gelu); old zcheck2 was buggy.
- core step exact (h 5e-5, out 5e-7): ACTMAP['core.']='gelu' (all core ffn + core.age gelu), M INCLUDES current e,
  ages in SECONDS, age emb ADDED TO MEMORY VECTORS (age_mode='val': mha(xq, Mt+ae)). H=4. Use run_stay_steps(s,opt) pattern,
  opt=dict(m_incl=True,au='s',am='val'). enc_eu2(atoms)->(e,u) exact. Xs=states fwd_2 (100 batch1 stays), row_start[s], BUN[s].
""")
hn = [n for n in NAMES if not n.startswith('core.') and not n.startswith('encode') and not n.startswith('embed.')]
print([(n, tuple(SHAPES[n])) for n in hn if not n.startswith('z_')])
print([ (n,tuple(SHAPES[n])) for n in NAMES if n.startswith('embed.') ])
```