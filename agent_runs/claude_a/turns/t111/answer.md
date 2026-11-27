The listing was cut off. One function, `tf_states`, has a very long default argument, and its printout used up the output limit. The functions themselves are all there, including:
- `rollout_stay` and `rollout_gen`: the offline dose-edit rollout;
- `tf_predict` and `tf_views`: teacher-forced predictions with `mods`;
- `step_b`: the core step;
- `head_value`: the value head;
- `prep_stay`, `stay_inputs` and `rates_by_bundle`.

So my offline machinery survived the restart. I still need the variable inventory, especially the parameter dict `P` the replica functions take and the stay lists. I also need the tensor names and shapes, and some sign of how run 105 computed the factual NLL.

**Next step:** print the same inventory compactly, with parameter names only and no default values. This run only collects information; it succeeds if `P`-like dicts, stay selections and data frames show up.

```python
import inspect, types, numpy as np, os
g = dict(globals())
fl = []
for k,v in g.items():
    if k.startswith('_') or not isinstance(v, types.FunctionType): continue
    try: ps = list(inspect.signature(v).parameters)
    except Exception: ps = ['?']
    fl.append(f"{k}({','.join(ps)})")
print("FUNCS:", " ".join(fl)[:2500])
vars_ = []
for k,v in g.items():
    if k.startswith('_') or isinstance(v,(types.FunctionType, types.ModuleType, type, types.BuiltinFunctionType)): continue
    if k in ('In','Out','exit','quit','get_ipython'): continue
    try:
        if isinstance(v, np.ndarray): s=f"{k}:nd{v.shape}"
        elif hasattr(v,'shape') and hasattr(v,'dtype'): s=f"{k}:T{tuple(v.shape)}"
        elif hasattr(v,'columns'): s=f"{k}:DF{v.shape}"
        elif isinstance(v,dict): s=f"{k}:dict[{len(v)}]{list(v.keys())[:3]}"
        elif isinstance(v,(list,tuple,set)): s=f"{k}:{type(v).__name__}[{len(v)}]"
        elif isinstance(v,(int,float,str,bool,np.floating,np.integer)): s=f"{k}={str(v)[:25]}"
        else: s=f"{k}:{type(v).__name__}"
    except Exception: s=f"{k}:?"
    vars_.append(s)
print("\nVARS:", "; ".join(vars_)[:5000])
print("\n/work/edited:", os.listdir('/work/edited')[:10], " fwd_4:", os.listdir('/work/fwd_4'))
dc = np.load('/work/dimcost.npy'); print('dimcost', dc.shape, 'lowest', np.round(np.sort(dc.ravel())[:12],5))
```