The last run's check matched a reimplementation of the model's state against the tool's `states` output. The error was about 5e-7 on every bundle, and no bundle was off by more than 1e-4. So in the runs after my notebook's last entry (48–63), I seem to have built a working CPU replica of the full core. That would let me test weight edits offline without spending forward passes. My notebook has no record of how I did it, or of what forward 4 was.

Next I will list the kernel's functions and variables and the contents of /work/fwd_*. If the cell history is available, I will also print the last cells, so I can reconstruct the replica and log it.

If that shows a function computing states and predicted means from the weights, I will extend it to an offline `dose_edit`-style rollout. That would let me measure the model's teacher-forced dose sensitivities before designing an edit. If no such function exists, the "err" result was something narrower and I will need to re-check it.

```python
import inspect, os, types
gl = dict(globals())
print('has In:', 'In' in gl, (len(gl['In']) if 'In' in gl else None))
funcs = sorted(k for k,v in gl.items() if isinstance(v, types.FunctionType) and not k.startswith('_'))
print("FUNCS:")
for k in funcs:
    try: sig = str(inspect.signature(gl[k]))
    except Exception: sig='?'
    print("  ", k, sig)
skip = {'In','Out','exit','quit','get_ipython'}
others = {k:(type(v).__name__, (getattr(v,'shape',None) if hasattr(v,'shape') else (len(v) if hasattr(v,'__len__') else ''))) for k,v in gl.items()
          if not k.startswith('_') and k not in skip and not isinstance(v,(types.FunctionType, types.ModuleType, type))}
print("VARS:")
for k in sorted(others): print("  ", k, others[k])
for d in sorted(os.listdir('/work')):
    p = os.path.join('/work',d)
    if os.path.isdir(p): print(d, sorted(os.listdir(p)))
if 'In' in gl:
    for i in range(max(1,len(In)-4), len(In)-1):
        print(f"----- In[{i}] -----"); print(In[i][:2500])
```