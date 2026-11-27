The input section shows that drug doses enter the core only through u_k, the action token. They do not enter the memory M, which holds only measurement bundles. So the only route from dose to state is the prelude's attention from the 4 slot tokens to the u_k token.

That explains why the state carries the dose so weakly. A strong, clean dose signal needs an edit along this route: action encoder → prelude attention value/output → b_in into h → read by a head unit.

Before designing it I need the replica's building blocks. I'll check:
- the signatures and referenced names of `step`, `mha`, `tlayer`, `bvd` and `zpost`, so I can test weight edits by changing the weight dict W;
- the number of attention heads;
- the encode_action and z tensor names.

If `mha` and `tlayer` take a weight-prefix argument, I can measure per-head attention to the u token and inject a rank-1 dose path. If not, I'll reimplement the prelude from the architecture text.

```python
import inspect, json, os
def sig(f):
    c=f.__code__; return f"{f.__name__}{inspect.signature(f)} names={list(c.co_names)[:14]}"
for fn in ['step','mha','tlayer','block','ffn','bvd','zpost','encode','run_cf']:
    try: print(sig(globals()[fn])[:300])
    except Exception as ex: print(fn,'ERR',str(ex)[:60])
o=[n for n in NAMES if not (n.startswith('core.') or n.startswith('embed') or n.startswith('attr_heads'))]
print(' '.join(f"{n}{tuple(SHAPES[n])}" for n in o)[:1500])
print([f"{i}:{l.strip()[:120]}" for i,l in enumerate(L) if 'head' in l.lower() and ('8' in l or '6' in l or '4' in l)][:6])
print({f:os.path.getsize('/work/'+f) for f in os.listdir('/work') if f.endswith('.py')})
```