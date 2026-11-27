The architecture description confirms one thing: dose information reaches the glucose and MAP predictions only through the recurrent state update (prelude, blocks, coda). The value head reads the state and a "last value" register of the same code, with no direct dose input. Any edit therefore has to put a dose signal, centred on what the state already expects, into the state and then read it out.

Before choosing where to edit, I'll check which helper functions and arrays from earlier runs are still in the kernel, and what forward pass 4 contained. That decides whether to resume the numpy re-implementation of a model step, which would let me test edits offline instead of spending forward passes.

```python
import os, types
g=[k for k,v in globals().items() if not k.startswith('_') and not isinstance(v,types.ModuleType)]
fn=[k for k in g if callable(globals()[k])]
arr=[k for k in g if hasattr(globals()[k],'shape')]
print('funcs:',fn[:60])
print('arrays:',[(k,tuple(globals()[k].shape)) for k in arr][:40])
print('fwd_4:',os.listdir('/work/fwd_4'))
```