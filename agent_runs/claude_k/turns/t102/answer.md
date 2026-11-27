The architecture text came through. Its key point is that the drug atoms reach the reading heads only through the recurrent state (u_k → z_post and prelude → h → out → view → q). The heads have no direct dose input. Any fix therefore has to carry an exact, linear dose signal through the state, or read the dose information the state already holds.

Next I'll list the shapes of the core, action-encoder and value-head tensors. I'm looking for a near-linear path, for example a residual or decaying channel, that a dose feature could ride on. If the coda or h update has a residual or diagonal structure, a dedicated "dose memory" channel is feasible. If every step mixes all dimensions nonlinearly, I'll fall back to a probe of the existing state read in the value head.

```python
pref = ('core.', 'encode_action.', 'attr_heads.value_bin', 'z_head', 'query_norm', 'out_norm', 'last_numeric')
rows = [(n, tuple(SHAPES[n])) for n in NAMES if n.startswith(pref)]
print(len(rows))
print('; '.join(f"{n}{s}" for n, s in rows))
```