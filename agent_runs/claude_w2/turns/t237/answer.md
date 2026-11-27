The last output was cut by a content filter, so I'll keep outputs short from now on. I did recover the gist of the recent scripts. `oracle.py` and `linor.py` are local rollout hooks that tilt the glucose/MAP bin distributions by an exactly computed dose term: lag-only, policy-deviation, or linear in absolute dose lags plus policy-of-glucose lags. `resid.py` regresses the original model's factual errors on dose and policy lags. `dec.py` and `states_ins.py` collect states and decode dose from them. Run 227 found that only the encoder output `u` carries a counterfactual dose change near-linearly; `out` does not (slope 0.54).

The notebook stops at run 207, so the results of the linear-oracle and residual runs (208–226) are unrecorded. Before designing the weight edit I need to see which kernel variables still exist and which results were stored. I'll print a compact inventory.

```python
import types
gl = {k:v for k,v in globals().items() if not k.startswith('_')}
fn = sorted([k for k,v in gl.items() if isinstance(v, types.FunctionType)])
print('FUNCS', len(fn), ' '.join(fn)[:1500])
def desc(v):
    if isinstance(v, np.ndarray): return 'nd'+str(v.shape)
    if torch.is_tensor(v): return 't'+str(tuple(v.shape))
    if isinstance(v, pd.DataFrame): return 'df'+str(v.shape)
    if isinstance(v, dict): return 'dict%d'%len(v)
    if isinstance(v, (list,tuple)): return 'list%d'%len(v)
    if isinstance(v,(int,float,str)): return repr(v)[:30]
    return None
items = [(k,desc(v)) for k,v in gl.items() if desc(v) is not None and k not in ('In','Out')]
print('VARS', ' | '.join(f'{k}:{d}' for k,d in sorted(items))[:3500])
```