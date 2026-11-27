The value head is an MLP (192→192→17), z_prior is an MLP and z_head is linear, and the coda is a residual transformer layer applied after a full linear map `c_out`. So every path from the state to the glucose logits passes through LayerNorm or GELU. No single rank-1 linear edit gives a clean dose readout.

The dose information has to be carried in the recurrent state from the drug bundle to later readings. It is not stored in the memory M, which holds only measurement encodings, so a precise mechanistic edit needs more care. Before designing it, I want to see what the kernel already holds: helper functions, targets per stay-hour, and the policy model.

This step is an inventory; it confirms nothing yet. Next I'll compare the original model's per-lag response in fwd_1 with the true per-lag kernel effect. If the model's response has the same time shape scaled down, a uniform amplification of the dose-deviation signal would work. If the shape differs, I need a lag-specific edit.

```python
import types
g = {k:v for k,v in globals().items() if not k.startswith('_')}
fn = sorted([k for k,v in g.items() if callable(v) and getattr(v,'__module__',None) in (None,'__main__') and not isinstance(v,type)])
print('FUNCS', fn)
var = []
for k,v in g.items():
    if k in fn or isinstance(v, types.ModuleType) or k in ('In','Out','get_ipython','exit','quit'): continue
    s = type(v).__name__
    if hasattr(v,'shape'): s += str(tuple(v.shape))
    elif isinstance(v,(list,dict,tuple)): s += f'[{len(v)}]'
    var.append(f'{k}:{s}')
print('VARS', ' | '.join(var)[:3500])
```