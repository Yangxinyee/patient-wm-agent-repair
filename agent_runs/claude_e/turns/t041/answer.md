Run 38 compared settings of my local re-implementation of the model against the tool's saved states. The best error was 0.257, so the re-implementation does not match the real model yet. The conversation was restarted, so I don't remember the architecture details or the code I wrote. Next I will re-read `/model/ARCHITECTURE.md` and list the kernel's variables and functions. That tells me what I have built and what is still unknown. If the listing shows a forward function and its configuration options, I can continue the matching. If not, I will rebuild it from the document.

```python
import inspect, types
print(open('/model/ARCHITECTURE.md').read())
print("=====GLOBALS=====")
for k,v in list(globals().items()):
    if k.startswith('_') or isinstance(v, types.ModuleType): continue
    t = type(v).__name__
    extra = ''
    try:
        if hasattr(v,'shape'): extra = str(tuple(v.shape))
        elif isinstance(v,(list,dict,tuple,set)): extra = f'len={len(v)}'
    except Exception: pass
    print(k, t, extra)
```