The bundle structure is clear. `BUN[stay]` is a list of (minute, events) pairs, and it lines up one-to-one with the rows of the states file (54 bundles and 54 rows for the first stay). The constants in `ctx_of` confirm the context formula: out + out_norm(mean of the time-flowed slots) + gap(log(dt)/10).

The remaining unknowns for the value head are these:
- the layout of the 8 `last_numeric` inputs;
- which `type_emb` row the value query uses;
- the `last_status` index;
- which activation the value head uses.

The column norms of the weight matrices will show which of the 8 inputs are actually used. I will also check the existing `fbin`/`zsc` helpers and the vocab keys. Only short summaries are printed.

```python
lnw = load('last_numeric.weight').numpy()
print('last_numeric col norms', np.round(np.linalg.norm(lnw, axis=0), 3))
enw = load('embed.numeric.weight').numpy()
print('embed.numeric col norms', np.round(np.linalg.norm(enw, axis=0), 3))
print('type_emb row norms', np.round(np.linalg.norm(load('type_emb.weight').numpy(), axis=1), 2))
print('last_status row norms', np.round(np.linalg.norm(load('last_status.weight').numpy(), axis=1), 2))
print('fbin', inspect.signature(fbin), 'zsc', inspect.signature(zsc) if callable(zsc) else type(zsc))
print('VOCAB keys', list(VOCAB.keys()))
for k_ in VOCAB:
    if k_ not in ('config','codes','event_kinds','atom_kinds','numeric_note','value_bin_centres','value_bin_note'):
        v_ = VOCAB[k_]
        print(k_, type(v_).__name__, list(v_.keys())[:12] if isinstance(v_, dict) else str(v_)[:200])
```