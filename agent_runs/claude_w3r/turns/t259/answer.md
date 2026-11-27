The last run printed the kernel's saved variables, then crashed on a listing bug. The useful parts survived: the weights, the cached prelude intermediates for 5,201 bundles (P_), the drug-rate targets (T_comb) and the candidate injection directions (cands). That is enough to build the channel. Before spending runs on the long prelude → b_in → head path, I will check one shortcut. I will grep ARCHITECTURE.md for how the reading head's query is formed (view, query, pending, filter). If a running infusion's rate enters that query directly, a much shorter path exists. If it does not, I keep design (alpha). I will also list the helper functions properly and start a session-11 entry in the notes.

```python
import re, types
A=open('/model/ARCHITECTURE.md').read()
for i,l in enumerate(A.splitlines()):
    if re.search(r'pending|filter|query|with_code|view|last_numeric', l, re.I): print(i, l.strip()[:200])
fn=sorted(k for k,v in list(globals().items()) if isinstance(v, types.FunctionType) and not k.startswith('_'))
print(len(fn), fn)
N=open('/work/notes.md').read(); print(len(N), [l[:60] for l in N.splitlines() if l.startswith('#')])
with open('/work/notes.md','a') as f:
    f.write("\n## Session 11 (restart 11)\n- Run 238: grep ARCH for query/pending path; list helper fns. Budget: 62 runs, 47 fwd left -> BUILD NOW.\n")
```