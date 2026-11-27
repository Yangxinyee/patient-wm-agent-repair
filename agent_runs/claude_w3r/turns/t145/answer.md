The architecture text confirms two things. The readout is the head's predicted mean (sum of p(b)·centre[b]), edited minus factual. The head's view includes `out` plus `out_norm` of the time-decayed h, and passes through `query_norm` before the value MLP. So a dose channel injected into h has to survive into `out` and into the decayed-h term.

Next I'll recall how my local implementation in /work/impl3.py is organised: function signatures, plus the shapes of a few stored arrays. The aim is to compute E at drug bundles and probe for a clean rate direction there. This is a lookup step, not a test.

```python
src = open('/work/impl3.py').read()
for line in src.splitlines():
    if line.startswith('def ') or line.startswith('# '):
        print(line[:150])
for k in ['E','E0','E1','XE','XU','JE','Zall','H','H2','X','Q','R','DEV','Ins','Gf','S_list','PA','OPTS']:
    v = globals().get(k)
    try: print(k, type(v).__name__, getattr(v,'shape',None) if hasattr(v,'shape') else (len(v) if hasattr(v,'__len__') else v))
    except Exception as ex: print(k, ex)
```