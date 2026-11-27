The records suggest nonzero drug effects, while the original rollouts respond only weakly. The architecture confirms that doses can affect the recurrent state, but does not show whether the proposed correction will remain stable. I’ll recover the saved probe functions and directions so I can test finite perturbations rather than rely on Jacobians alone.

```python
import inspect
gg = list(globals().items())
print("=== FUNCTIONS ===")
for k, v in gg:
    if inspect.isfunction(v) and not k.startswith("_"):
        print(k, str(inspect.signature(v))[:180])
print("\n=== RECENT ARRAYS ===")
aa = [(k,v) for k,v in gg if isinstance(v, (np.ndarray, torch.Tensor)) and not k.startswith("_")]
for k,v in aa[-55:]:
    print(k, tuple(v.shape), str(v.dtype))
print("\n=== RECENT CONTAINERS ===")
cc = [(k,v) for k,v in gg if isinstance(v, (dict,list,tuple)) and not k.startswith("_")]
for k,v in cc[-35:]:
    print(k, type(v).__name__, len(v), str(list(v)[:5])[:180])
print("\n=== OTHER RECENT VARIABLES ===")
for k,v in gg[-70:]:
    if isinstance(v,(str,float,int,np.number)):
        print(k, str(v)[:180])
print("EDITS", edited())
```
