The scaffold functions are clear now. `make_scaffold` and `scaffold_v` take the 8 role dims (`JALL`) and cut their reads and writes everywhere:
- prelude and coda attention and FFN,
- the slots, prelude_norm, b_in and c_out,
- out_norm, gap, z_head, type_emb and query_norm,
- the value-MLP layers.

`scaffold_v` then wires specific paths through `flags`.

`dimcost` (3 rows) is the factual-NLL change from killing each single dim. The mean is about +0.006, but the chosen role dims cost about −0.01 to +0.001, i.e. near-free.

The obstacle is that I can't see the source of what I built, and rebuilding blind is costly. If the kernel is IPython-like, the full input history may still be there. Recovering it would give me the exact code of runs 143–153 and let me save it to a file for future restarts. If no history exists, I'll re-implement the edit as explicit code in /work/edit_lib.py.

```python
import linecache, sys
cands = [k for k in ['In','_ih','_i','_ii','_iii','Out','_oh'] if k in globals()]
print('history vars:', cands)
for k in cands:
    v=globals()[k]
    try: print(k, type(v).__name__, len(v))
    except Exception: print(k, type(v).__name__)
print('linecache keys sample:', [k for k in linecache.cache.keys()][:10], len(linecache.cache))
# search any string-containing containers for the def
hits=[]
for k,v in list(globals().items()):
    if isinstance(v,str) and 'def scaffold_v' in v: hits.append(k)
    if isinstance(v,(list,tuple)) and v and isinstance(v[0],str):
        if any(('def scaffold_v' in s) for s in v if isinstance(s,str)): hits.append(k)
print('hits', hits)
try:
    ip=get_ipython(); print('ipython', type(ip).__name__, len(ip.user_ns.get('In',[])))
except Exception as ex: print('no ipython', ex)
print([f for f in os.listdir('/work')], os.path.getsize('/work/notes.md'))
```