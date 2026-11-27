The kernel still holds the full replica toolkit: `run_stay`, `core_step`, the head replicas, `gen_samples` and `fit_dec`. `samp_pert` is a list. Next I need the call signatures and whether the source code is recoverable, so I can reuse these functions instead of rewriting them. I'll print signatures for the key functions, the structure of `samp_pert[0]`, and the start of `gen_samples`' source, all briefly.

```python
import inspect
keyf = ['run_stay','run_head','head_value2','head_P2','gen_samples','fit_dec','eval3','true_shift','truth_rows','ne_expect','base_ins','ctrl_ins','ctrl_ne','add_pol','bundles_of_stay','core_step','eval_head','wc_vec']
for f in keyf:
    try:
        print(f, str(inspect.signature(g[f]))[:120])
    except Exception as e:
        print(f, 'ERR', e)
print('samp_pert len', len(sp), type(sp[0]).__name__)
s0 = sp[0]
if isinstance(s0, dict):
    print({k: (type(v).__name__, getattr(v, 'shape', None)) for k, v in s0.items()})
elif isinstance(s0, (tuple, list)):
    print([(type(v).__name__, getattr(v, 'shape', None)) for v in s0])
try:
    src = inspect.getsource(g['gen_samples'])
    print('gen_samples src len', len(src)); print(src[:1200])
except Exception as e:
    print('no src', e)
```