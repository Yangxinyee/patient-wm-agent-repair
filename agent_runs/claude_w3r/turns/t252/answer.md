The impl files only hold data-preparation code. The model simulator lives in kernel functions without source, such as `core_step_v`, `run_from`, `step_at`, `prelude_att`, `head_inputs`, `pred_mean_gf`, `dnll` and `att_edit`. My array listing failed because some globals have a non-array `.shape`.

Next I will print each function's signature and the global names it uses, which can be read from its code object. I will also list the real arrays, tensors, DataFrames and dicts. That tells me how to call the simulator for the edit, in particular whether `att_edit` and `run_from` already accept edited weights. If the signatures show that functions read the weight dict `Wt`, I can test edits locally by temporarily swapping entries.

```python
import inspect, numpy as np, torch, pandas as pd
out=[]
for k in fns:
    v=globals()[k]
    try: sig=str(inspect.signature(v))
    except Exception: sig='(?)'
    c=getattr(v,'__code__',None)
    gl=[n for n in (c.co_names if c else []) if n in globals() and not callable(globals()[n])][:12]
    out.append(f"{k}{sig[:70]} G:{','.join(gl)}")
print('\n'.join(out))
A=[]
for k,v in globals().items():
    if k.startswith('_'): continue
    if isinstance(v,(np.ndarray,torch.Tensor)): A.append(f"{k}{tuple(v.shape)}")
    elif isinstance(v,pd.DataFrame): A.append(f"{k}[DF{v.shape}]")
    elif isinstance(v,dict) and len(v)>0: A.append(f"{k}{{dict {len(v)}}}")
print('VARS:', '; '.join(sorted(A))[:6000])
```