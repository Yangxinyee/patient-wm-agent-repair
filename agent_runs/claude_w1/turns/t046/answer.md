`inspect.getsource` fails here because the kernel runs code without source files. I'll read the helper signatures with `inspect.signature`, which doesn't need source, and print the remaining tensor shapes.

```python
import inspect
for fn in ['panel','block_stats','ols','base_ins','lp','lp_series','deconv','ctrl_ins','ctrl_ne','rd','true_shift','lin','LN','mlp','encode_set','atom_vec']:
    f_=globals()[fn]
    print(fn, str(inspect.signature(f_))[:120], '| locals:', f_.__code__.co_varnames[:12])
keys=['embed.bins','embed.categorical','encode_action','z_prior','gap.','last_numeric','type_emb','embed.code','last_status','z_post','z_in']
print('; '.join(f"{n}:{tuple(SHAPES[n])}" for n in NAMES if any(n.startswith(q) for q in keys)))
print(list(VOCAB.keys()), VOCAB.get('config'))
```