The architecture confirms that the edit must influence the recurrent state before it can affect later readings. The previous diagnostic output was truncated, so I’ll inspect only the recent edit objects and helper names. This should reveal whether the dose signal reaches the posterior at the intended scale.

```python
import re
_g129 = list(globals().items())
_recent129 = [(k,v) for k,v in _g129
              if re.search(r"(12[0-7])(?:_|$)", k) and not k.startswith("_")]
print("=== RECENT EDIT OBJECTS ===")
for k,v in _recent129:
    if isinstance(v, (np.ndarray, torch.Tensor)):
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, dict):
        print(k, "dict keys:", list(v)[:15])
    elif isinstance(v, (list,tuple)):
        print(k, type(v).__name__, len(v), repr(v[:4])[:160])
    elif isinstance(v, str):
        print(k, "str length", len(v), repr(v[:90]))
    elif isinstance(v, (int,float,bool,np.number)):
        print(k, repr(v))
    else:
        print(k, type(v).__name__)
print("=== LOCAL FUNCTIONS ===")
print([k for k,v in _g129 if callable(v) and getattr(v,"__module__","") == "__main__"])
print("=== OTHER RELEVANT NAMES ===")
print([k for k,v in _g129 if not k.startswith("_") and
       any(t in k.lower() for t in ("post","policy","innov","direction","decoder","gain","action","bundle","latent"))
       and (k,v) not in []])
print("=== POSSIBLE HISTORY ===")
print([k for k,v in _g129 if k in ("In","_ih","history","get_ipython","python_history","run_history")])
print("=== POSTERIOR EDIT SHAPES ===")
for n in edited():
    print(n, tuple(load(n).shape))
```
