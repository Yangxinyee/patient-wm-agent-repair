The kernel holds a lot of state. The newest files are `s1data.npz` (15:10) and `jac.npz` (15:20); `inj.py` (12:14) and `tf_train.npz` (4,000 train stays run teacher-forced through my replica) came earlier. So after run 177 I built a z-injection step and teacher-forced data, then computed per-bundle Jacobians. I need to know what those objects mean before going on. Next I will print `inj.py` and compact descriptions (type, shape, a few values) of the most recent kernel variables. If they hold fitted residual-channel directions, gains or decoders, I will resume Stage 1 from there. If not, I will rebuild it from `tf_train.npz`.

```python
import types
print(open('/work/inj.py').read())
def desc(v):
    if isinstance(v,np.ndarray): 
        s=f'arr{v.shape}'
        if v.size<=6: s+=str(np.round(v.astype(float),3).tolist())
        return s
    if isinstance(v,dict): return f'dict{len(v)}:{list(v.keys())[:6]}'
    if isinstance(v,(list,tuple)): return f'{type(v).__name__}{len(v)}'+(f'[{type(v[0]).__name__}]' if len(v) else '')
    if isinstance(v,types.FunctionType): return 'fn('+','.join(v.__code__.co_varnames[:v.__code__.co_argcount])+')'
    if isinstance(v,(int,float,np.floating)): return f'{float(v):.4g}'
    return type(v).__name__
print(' ; '.join(f'{k}={desc(globals()[k])}' for k in g[-120:]))
```