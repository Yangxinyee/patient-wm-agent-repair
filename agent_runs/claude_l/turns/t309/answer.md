Picking up from the restart. Run 281 checked the extra-token approach I planned for the edit. It showed that drug bundles' attention heads 0–2 look almost only at the action token u, while head 3 looks at the slots. It also showed that the dose directions I planned to inject (`VD_`) lie almost entirely (99.5%) inside the subspace that changes attention scores. So adding the dose signal along those directions would also move the attention split, not just the values passed through it. That pushes me toward building the injected direction outside the attention-sensitive subspace, while keeping its effect on the values.

I have only 19 Python runs left (45 forward passes), and the cell sources were lost in the last restart. So the next step is a cheap inventory of what is still in the kernel and in /work: functions, signatures, saved arrays, and which builder file holds the injection code. I'll also write the run-281 findings to the notebook so they survive the next restart. If the builder (`mkW3`), the offline rollout evaluator (`evalW`) and the teacher-forced NLL (`tfnll`) are still usable, I can tune the gains offline and only use forward passes to confirm. If they're broken, I'll rebuild a minimal version from the /work/*.py files instead.

```python
import inspect, types, os, re, numpy as np
with open('/work/notes.md','a') as f:
    f.write("\n- Run 281: drug bundles: heads0-2 attend u (~.94-1.0), head3 slots. Attn-logit grad wrt token: 99.9% energy in top16 dirs. "
            "VD_ (ins norm 3.6, ne 9.7) lies 99.5% in that key-sensitive subspace (|dlogit| 2.75 ins, 6.85 ne per unit) -> injecting along VD_ shifts attention split; "
            "need direction with ~0 key effect but same value effect (heads0-2).\n")
g = dict(globals())
funcs = {k:v for k,v in g.items() if isinstance(v, types.FunctionType) and not k.startswith('_')}
print("FUNCS", len(funcs))
for k,v in sorted(funcs.items()):
    try: sig = str(inspect.signature(v))
    except Exception: sig='?'
    try: src = inspect.getsourcefile(v)
    except Exception: src='?'
    print(f"  {k}{sig[:110]} [{os.path.basename(str(src))}]")
arrs = sorted((k, tuple(v.shape)) for k,v in g.items() if isinstance(v,np.ndarray) and not k.startswith('_'))
print("ARRAYS", arrs[:120])
oth = sorted((k,type(v).__name__, (len(v) if hasattr(v,'__len__') else '')) for k,v in g.items()
             if not k.startswith('_') and not isinstance(v,(types.FunctionType,types.ModuleType,np.ndarray,type)) and k not in ('In','Out','g','funcs','arrs'))
print("OTHERS", oth[:150])
for fn in sorted(os.listdir('/work')):
    p='/work/'+fn
    if fn.endswith('.py'):
        s=open(p).read(); print(fn, len(s), 'defs:', re.findall(r'^def (\w+)', s, re.M))
print("edited:", edited())
```