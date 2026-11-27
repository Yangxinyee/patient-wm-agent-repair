Run 217 compared candidate directions in the model's hidden state (top eigenvector, singular vectors, "consistent" and tangent directions) as possible carriers of a lasting dose memory. For each it printed the spread of the glucose or MAP response gain across bundles, the noise, and the size of the mean Jacobian times the direction. The NE eig1 direction stands out: gain spread 0.81 against noise 0.16. The insulin directions are weaker, about 0.35 against 0.17.

Runs 177–217 are not in my notebook, so I have lost the exact meaning of those numbers and of the current edit state. The next three replies were blocked because their output was too long.

Next I will look at the kernel's variables, the helper files and the saved edit, printing only short summaries. I also want to confirm my working hypothesis:
- the only active edit is z_head.weight set to 0, which slightly improves the value NLL;
- the z_post / z_in "residual dose" channel has been designed but not yet saved.

If I find an unsaved or saved z_post/z_in edit, or a dose-memory direction already fitted, that hypothesis is wrong and I will pick up from there.

```python
import types, os
g = {k: v for k, v in globals().items() if not k.startswith('_') and not isinstance(v, types.ModuleType)}
funcs = sorted(k for k, v in g.items() if isinstance(v, types.FunctionType))
arrs = sorted(k for k, v in g.items() if hasattr(v, 'shape') and not isinstance(v, type))
oth = sorted(k for k in g if k not in funcs and k not in arrs and not isinstance(g[k], type))
print('nfuncs', len(funcs)); print(' '.join(funcs)[:1200])
print('narrays', len(arrs)); print(' '.join(f"{k}{tuple(g[k].shape)}" for k in arrs)[:1200])
print('nother', len(oth)); print(' '.join(oth)[:800])
print('edited:', edited())
w = load('z_head.weight'); print('z_head.weight absmax', float(abs(w).max()))
for f in ['inj.py', 'wm2.py', 'wm2_rollout.py', 'tf.py', 'setup.py']:
    src = open('/work/' + f).read()
    defs = [l[4:].split('(')[0] for l in src.splitlines() if l.startswith('def ')]
    print(f, len(src), '|', ' '.join(defs)[:300])
print(os.listdir('/work/edited'), os.path.getsize('/work/notes.md'))
```