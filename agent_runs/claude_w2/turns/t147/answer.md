The core step now matches the tool's states to float precision: mean abs error 1e-5 on `out` and 4e-4 on h. The working definition:
- the memory M includes the current bundle's e;
- age is in seconds, and `core.age(log1p(age))` is added to the memory vectors before the key/value projection (so it enters both keys and values);
- u for bundles without a drug is the encoder's "empty set" vector, not zero.

The run-133 mismatch was a row-alignment problem plus these details. I'll record this in the notebook now, so a restart doesn't lose it. Then I'll look at the head tensors (type_emb, last_numeric, last_status, attr_heads.value_bin) and the factual output I have for the same stays. The next step is reproducing the value head's predicted means from the exact states.

```python
notes_add = r'''
## Restart 5 (runs 134-141): EXACT CORE STEP FOUND (code in /work/core_test.py, needs impl.py exec'd first)
- fwd_3 states rows are NOT sorted like my bundle table: align by (stay_id, round(hour*60)) -> SxA = Sx[perm_S].
- core step exact (out err 1e-5, h 4e-4): memory M = e of ALL bundles of the stay up to AND INCLUDING current (last 128),
  age = (t_now - t_j) in SECONDS, agee = core.age(log1p(age)) ADDED TO M BEFORE kv projection (age_mode 'inadd': keys and values),
  u of bundles w/o drug = encoder empty-set vector (NOT zero). ctx gap = gap(log(dt_s)/10). z = posterior mean (first 32 of z_post).
- Lesson: kernel functions have no source -> keep all code in /work/*.py files.
'''
open('/work/notes.md','a').write(notes_add)
hd=[(n,SHAPES[n]) for n in NAMES if n.split('.')[0] in ('attr_heads','type_emb','last_numeric','last_status','query_norm','embed_transition','z_head','z_prior')]
print(hd)
print(fa4.columns.tolist()); print(fa4.head(3).T)
print([k for k in V.keys()], V.get('types', None) if 'types' in V else '')
```