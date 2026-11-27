Nothing was run last time, so the state is unchanged. The core-stage edits are built only in the kernel (`EDIT` dict); nothing is saved yet. The last stage is still missing: the value head has to read the added dose signal from q and turn it into a mean shift. That shift must be gated so it applies only to glucose_finger (insulin) and map_arterial (norepinephrine). Only 25 Python runs are left. So this step only checks the helper signatures and the shapes of the stored value-head objects, to avoid failed calls later. The step works if every signature and shape prints, which lets the value-head stage be written in the next run.

```python
import inspect, re
with open('/work/notes.md','a') as f:
    f.write("\n## Session 14 (restart 14)\n- Run 276: inspect helper signatures + value-head tensors before building value-head stage. Keep replies short.\n")
for fn in ['wc_of','wc_full','pick','T32','LNn','apply_local','prelude_out','wc_row','step_at','stay_data','run_from','coda','u_of']:
    g = globals().get(fn)
    try: print(fn, inspect.signature(g))
    except Exception as e: print(fn, 'ERR', e)
for v in ['DQ','RD','GD','DECL','DOUT','dec_b','EDIT','TY']:
    obj = globals().get(v)
    if isinstance(obj, dict):
        print(v, {k:(tuple(x.shape) if hasattr(x,'shape') else x) for k,x in obj.items()})
for v in ['Dm','Ec','Rsp','Pq','TIDX']:
    x = globals().get(v); print(v, getattr(x,'shape',None), getattr(x,'dtype',None))
for n in NAMES:
    if any(s in n for s in ['value_bin','embed_transition','query_norm','embed.code']):
        print(n, SHAPES[n])
A = open('/model/ARCHITECTURE.md').read()
for i,m in enumerate(re.finditer(r'(?i)(gelu|relu|silu|mlp\()', A)):
    if i>=5: break
    s=max(0,m.start()-120); print('ARCH:', A[s:m.end()+80].replace('\n',' '))
print('VOCAB codes:', VOCAB.get('codes', None) if isinstance(VOCAB,dict) else None)
```