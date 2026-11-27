Run 243 wasted most of its output. `mk_zfn` has a default argument that holds the entire weight dictionary, so printing its signature dumped the tensors and cut off the rest. That tells me the helper takes a mode and a weight dict. It does not show whether the copy-of-u plus residual design is already built.

Next I'll print only the argument names and the first part of each helper's source, with no signatures, so the output stays short. If the sources show the z-token construction and the residual map, I'll go straight to fitting and offline testing. If not, I'll write the construction myself in the following run.

```python
import inspect
for nm in ['mk_zfn','make_zfn','copyz','ztok_orig','rollout','true_shift','u_for','u_with','u_rate','zA','zB','zC']:
    f = globals().get(nm)
    co = getattr(f, '__code__', None)
    args = co.co_varnames[:co.co_argcount] if co else '?'
    try: s = inspect.getsource(f)
    except Exception: s = '(no source)'
    print('=====', nm, args, len(s)); print(s[:900])
```