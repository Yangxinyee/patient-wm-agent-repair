The architecture confirms that dose effects must pass through the latent, prelude and recurrent state. I’ll inspect the retained transport helpers and compensation fit next; identifying their exact inputs and outputs is necessary before testing where the half-dose response disappears.

```python
import dis
print("=== RECENT OBJECTS ===")
for n, x in sorted(list(globals().items())):
    if n.endswith(('218','219','220','221')) or any(s in n.lower() for s in ['direction','restricted','refit','fit220']):
        if inspect.isfunction(x):
            print(n, "function", inspect.signature(x))
        elif isinstance(x, dict):
            print(n, "dict", list(x.keys())[:14])
        elif hasattr(x, 'shape'):
            print(n, type(x).__name__, tuple(x.shape))
        elif isinstance(x, (list,tuple)):
            print(n, type(x).__name__, len(x))
        else: print(n, type(x).__name__, str(x)[:100])
print("=== SOURCE STORAGE ===")
for n,x in sorted(list(globals().items())):
    if isinstance(x,str) and len(x)>250 and any(t in n.lower() for t in ['src','source','code','body','text']):
        print(n, len(x), x[:70].replace('\n',' '))
print("=== SOURCE INSPECTION HELPER ===")
print(dis.Bytecode(show182).dis()[:3200])
print("=== PRELUDE AND RECURRENT HELPERS ===")
for nm in ['prelude155','state_update82','response186']:
    fn223=globals()[nm]
    try: txt223=inspect.getsource(fn223)
    except Exception: txt223=dis.Bytecode(fn223).dis()
    print(nm, txt223[:2000])
```
