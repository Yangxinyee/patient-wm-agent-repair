The last result checked whether prelude head 2 could be made to attend the empty dose token at bundles with no dose. On those bundles the real tokens score 7 points higher than the empty token on average, and up to 20 points higher. The extra score I could add is 17 to 19 points. Some real tokens also project fully (1.0) onto that added direction. So redirecting attention is not reliable, and I am dropping it.

I will use the cleaner route instead. Head 2 has unused value dimensions, and I will carry the dose rate through those into a low-variance slot direction. A rank-1 change to the input map from the prelude output into the core state (`core.b_in`) then carries the signal into the state. Unused units in the value head turn it into a shift of the predicted mean.

Only 66 Python runs are left, and many earlier runs were never written down. So the next step is to list the helper code saved in /work/impl*.py and the kernel's arrays, so I can reuse the validated local simulator rather than rebuild it. If the files hold the core-step and head functions, I build the edit on them. If they don't, I rebuild only the minimal pieces.

```python
import re
for f in ['impl.py','impl2.py','impl3.py','impl4.py']:
    s = open('/work/'+f).read()
    defs = re.findall(r'^def (\w+)\(([^)]*)\)', s, re.M)
    print(f, len(s), 'chars:', ', '.join(f"{d[0]}({d[1][:40]})" for d in defs))
fns = sorted(k for k,v in globals().items() if callable(v) and not k.startswith('_') and type(v).__name__=='function')
print('FUNCS:', ', '.join(fns))
arr = sorted((k, tuple(v.shape)) for k,v in globals().items() if hasattr(v,'shape') and not k.startswith('_') and hasattr(v,'ndim'))
print('ARRAYS:', '; '.join(f"{k}{s}" for k,s in arr[:150]))
with open('/work/notes.md','a') as fh:
    fh.write("\n## Session 11 (restart 11)\n- Run 234: head2 no-dose score gap real-u0 mean 7 max 20 vs achievable boost 17-19 and real-token proj up to 1.0 -> attention redirect dropped.\n"
             "- Going with free head2 value dims -> low-var slot dir -> b_in rank-1 -> head dead units. Keep replies short.\n")
```