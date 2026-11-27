`atom_vec` uses the globals `T`, `code_id`, `torch`, `zeros`, `fbin` and `zsc`, and one of these is now a function where a subscriptable object was expected. `T` shows up as a function in my helper list, so it is the likely culprit. I will print their types and disassemble line 17 of `atom_vec` to see exactly what is subscripted, then restore that object. I need a working, verified `atom_vec` to build exact NE atom vectors for the gated correction.

```python
import dis, io, contextlib
for n in ['T','code_id','zeros','fbin','zsc']:
    v=globals().get(n); print(n, type(v), (v.__code__.co_varnames[:v.__code__.co_argcount] if callable(v) and hasattr(v,'__code__') else ''))
buf=io.StringIO()
with contextlib.redirect_stdout(buf): dis.dis(atom_vec)
L=buf.getvalue().split('\n'); idx=[i for i,l in enumerate(L) if l.strip().startswith('17 ') or l.startswith(' 17') or l.startswith('17')]
print('\n'.join(L[idx[0]:idx[0]+14]) if idx else '\n'.join(L[:60]))
```